/***************************************************************************                                                                                     
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
*                                                                                    
*   Licensed under the Apache License, Version 2.0 (the "License");   
*   you may not use this file except in compliance with the License.                 
*   You may obtain a copy of the License at                                          
*                                                                                    
*       http://www.apache.org/licenses/LICENSE-2.0                      
*                                                                                    
*   Unless required by applicable law or agreed to in writing, software              
*   distributed under the License is distributed on an "AS IS" BASIS,              
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
*   See the License for the specific language governing permissions and              
*   limitations under the License.                                                   

***************************************************************************/                                                                                     

//#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <vector>

#ifdef CPP_AMP
#include <amp.h>
#endif

#include <tbb/task_scheduler_init.h>

#include "matrix_utils.h"
#include "image_utils.h"

#include "hessian.h"
#include "hessian_tbb.h"
#include "myocl.h"
#include "hessian_ocl.h"
#include "hessian_oclcpp.h"
#include "hessian_boltcl.h"
#include "hessian_amp.h"
#include "hessian_amp_range.h"
#include "hessian_cpuserial.h"

#include <bolt/transform_reduce.h>    // for accessing bolt::computeUnits 

//unsigned p_hessianMode = HessianMode::e_None; 
//unsigned p_hessianMode = HessianMode::e_Tbb | HessianMode::e_OclCppGpu | HessianMode::e_OclCppGpuScalar;
//unsigned p_hessianMode = HessianMode::e_Tbb | HessianMode::e_CppAmp | HessianMode::e_BoltForAmp| HessianMode::e_BoltForAmpLambda;
//unsigned p_hessianMode = HessianMode::e_Tbb  | HessianMode::e_OclCppGpuScalar | HessianMode::e_BoltForAmp ;
//unsigned p_hessianMode = HessianMode::e_OclCppGpuScalarMatch2 | HessianMode::e_OclCppGpuScalarMatch |	HessianMode::e_CppAmpUseWorker  | HessianMode::e_CppAmp ;
//unsigned p_hessianMode =HessianMode::e_BoltForAmpLambda | HessianMode::e_CppAmp;
unsigned p_hessianMode =  HessianMode::e_OclGpuScalarMatch;

const char *p_fileName = "marina.bmp";
int p_iters = 10;
int p_computeUnits = 0;  // leave at 0 to match Bolt hard-coded constant.
int p_wgPerComputeUnit = 7; // Leave at 7 to match Bolt
int p_localW = 8;
int p_localH = 8;
int p_roiBorder=1;

std::string p_oclCompilerOpt = "";

int p_zeroCopy = 1;


void printH3( const H3& h, std::ostream& s, const char* varName )
{
	if( varName )
		s << varName << " = [" << std::endl;
	else
		s << "H = [" << std::endl;

	for(int i = 0; i<3; i++)
	{
		for(int j = 0; j<3; j++)
			s << h(i, j) << " "; 
		s << std::endl; 
	}
	s << "];" << std::endl;
}




void printAccelerator(const concurrency::accelerator &acc)
{
	using namespace std;
	wcout << "description: " << acc.description << endl;
	wcout << "is_debug = " << acc.is_debug << endl;
	wcout << "is_emulated = " << acc.is_emulated <<endl;
	wcout << "dedicated_memory = " << acc.dedicated_memory/1024.0 << "MB" << endl;
	wcout << "device_path = " << acc.device_path << endl;
	wcout << "has_display = " << acc.has_display << endl;                
	wcout << "version = " << (acc.version >> 16) << '.' << (acc.version & 0xFFFF) << endl;
};

void printAcceleratorView(const concurrency::accelerator_view &accView)
{
	printAccelerator(accView.accelerator);
	std::wcout << "queue_mode = " << (int)accView.queuing_mode << std::endl;

};



void printCppAmpAccelerators()
{
	std::wcout << "C++ AMP Accelerators:\n";
	auto accelerators = concurrency::accelerator::get_all();
	std::for_each(begin(accelerators), end(accelerators),[=](concurrency::accelerator acc){ 
		printAcceleratorView(acc.default_view);
		std::wcout << std::endl;
	});
}

void printOclDevice(cl::Device d)
{
	std::cout << d.getInfo<CL_DEVICE_NAME>()
		<< " CU="<< d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() 
		<< " Freq="<< d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "Mhz"
		<< " Mem="<< d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/1024.0/1024.0 << "MB"
		<< "\n";
};





__int64 StartProfile() {
	__int64 begin;
	QueryPerformanceCounter((LARGE_INTEGER*)(&begin));
	return begin;
};

void EndProfile(__int64 start, int numTests, std::string msg) {
	__int64 end, freq;
	QueryPerformanceCounter((LARGE_INTEGER*)(&end));
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	double duration = (end - start)/(double)(freq);
	printf("%s %6.2fs, numTests=%d %6.2fms/test\n", msg.c_str(), duration, numTests, duration*1000.0/numTests);
};

//======
// Warp-frame implementation:
void indexCorrection(H3& corrected, const H3& input, utils::Size inFrameSize, utils::Size outFrameSize) throw()
{
	/////////////////////////////////////////////////////////////////////////////////////////////
	// Following code does essentially this:
	// 1. Translate coordinate of the destination image to the center of the frame [dstXc, dstYc]
	// 2. warp with projection matrix
	// 3. translate from center- to index-relative coordinates of source frame.
	//
	// For Homogeneous Coordinate space this is equivalent to following matrix multiplication:
	// >>           m = T(srcXc, srcYc)*P(mvec)*T(-dstXc, -dstYc)              <<
	/////////////////////////////////////////////////////////////////////////////////////////////

	double dstXc = (double)(outFrameSize.width)  / 2.0 + 0.5;
	double dstYc = (double)(outFrameSize.height) / 2.0 + 0.5;
	double srcXc = (double)(inFrameSize.width)	 / 2.0 + 0.5;
	double srcYc = (double)(inFrameSize.height)  / 2.0 + 0.5;

	// calculate corrections for first translation
	double srcX =  -dstXc*input[0] -dstYc*input[1] + input[2];
	double srcY =  -dstXc*input[3] -dstYc*input[4] + input[5];
	double D	=  -dstXc*input[6] -dstYc*input[7] + input[8];

	corrected[0] = input[0] + srcXc*input[6];
	corrected[1] = input[1] + srcXc*input[7];

	corrected[3] = input[3] + srcYc*input[6];
	corrected[4] = input[4] + srcYc*input[7];

	corrected[6] = input[6];
	corrected[7] = input[7];

	corrected[2] = srcX + srcXc*D;
	corrected[5] = srcY + srcYc*D;
	corrected[8] = D;
}

inline float fNaN() {
	const long l = 0x7fffffff;
	return *(float*)&l;
};

__forceinline float interp_lin( const utils::Matrix<float>& I, float x, float y, const utils::Rect& clipRgn ) throw()
{
	int ix = (int)x;
	int iy = (int)y;

	if( (ix <  clipRgn.left)   | (iy < clipRgn.top) |
		(ix >  clipRgn.right)  | (iy > clipRgn.bottom ) )
		return fNaN();

	float dx = x - ix;
	float dy = y - iy;

	if(ix == clipRgn.right)  { ix = ix - 1; dx = dx+1.0f; }
	if(iy == clipRgn.bottom) { iy = iy - 1; dy = dy+1.0f; }


	const float* data1 = I[iy];
	const float* data2 = I[iy+1];

	return   (1.0f-dy)*( (1.0f-dx)*data1[ix] + dx*data1[ix+1] ) +
		dy *( (1.0f-dx)*data2[ix] + dx*data2[ix+1] );
}

static void warpFrame( utils::Matrix<float>& wI,
					  const utils::Matrix<float>& I,
					  const utils::Rect& stripe,
					  const H3& map,
					  const utils::Rect& clipRgn) throw()
{
	H3 m;
	indexCorrection( m, map, I.getSize(), wI.getSize() );

	__declspec(align(16)) float h[12];

	h[0] = (float)m[0]; h[1] = (float)m[1];  h[2]  = (float)m[2]; h[3]  = 0.0f;
	h[4] = (float)m[3]; h[5] = (float)m[4];  h[6]  = (float)m[5]; h[7]  = 0.0f;
	h[8] = (float)m[6]; h[9] = (float)m[7];  h[10] = (float)m[8]; h[11] = 0.0f;

	if( (m[6] == 0.0) && (m[7] == 0.0) && (m[8] == 1.0f))
	{
		for( long y = stripe.top; y <= stripe.bottom; y++ )
		{
			float* wi = wI[y];
			float srcX = y*h[1] + h[2];
			float srcY = y*h[5] + h[6];
			for( int x = stripe.left; x <= stripe.right; x++ )	
				wi[x] = interp_lin( I, srcX + x*h[0], srcY + x*h[4], clipRgn);
		}
	}
	else
	{
		for( long y = stripe.top; y <= stripe.bottom; y++ )
		{
			float* wi = wI[y];
			for( long x = stripe.left; x <= stripe.right; x++ )
			{ 
				float D    = 1.0f/(x*h[8] + y*h[9] + h[10]);
				float srcX = D   *(x*h[0] + y*h[1] + h[2]);
				float srcY = D   *(x*h[4] + y*h[5] + h[6]);
				wi[x] = interp_lin( I, srcX, srcY, clipRgn);
			}
		}
	}
}



void printLOC(const char *tag, const char *fileName, const char *regionName, int regionStart, int regionEnd)
{
	std::string baseFileName(fileName);
	size_t lastSlash = baseFileName.rfind("\\");
	baseFileName = baseFileName.substr(lastSlash+1);
	printf("%-10s, %-10s, %d, %s, %d, %d\n", tag, regionName, regionEnd - regionStart, baseFileName.c_str(), regionStart, regionEnd);
};


int sizeTempBufferOCL_Bytes(const cl::Device &d) 
{
	const int wgPerComputeUnit=7; // pick a number high enough to hide latency
	const int computeUnits = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	int elements     = wgPerComputeUnit * computeUnits * 16; 
	int sizeBytes = elements * sizeof(float);
	return sizeBytes;
};

int sizeTempBufferAMP_Elements() 
{
	const int wgPerComputeUnit=7; // pick a number high enough to hide latency
	const int computeUnits = 10;
	int elements     = wgPerComputeUnit * computeUnits * 16; 
	return elements;
};




void init(int argc, char* argv[]) {
	// usage: hessian [options]
	// See hessian.h for possible values for HESSIAN_MODE
	// All parms are optional.
	int p_printAccelerators = 1;


	int tbbThreads=0;
	int ampAcceleratorNum = -1;

	int argI=1;
	while (argI<argc && (argv[argI][0] == '-')) {
		std::string argS = argv[argI];

		int iValue = 1;
		double fValue =1.0;
		std::string sValue="";
		size_t equalPos = argS.find("=");
		if (equalPos != std::string::npos) {
			sValue = argS.substr(equalPos+1);
			argS   = argS.substr(0, equalPos);
			iValue = strtoul(sValue.c_str(), NULL, 0);
			fValue = atof(sValue.c_str());
			printf ("arg=%s s=%s i=%d f=%6.2f\n", argS.c_str(), sValue.c_str(), iValue, fValue);
		}
		bool bValue = iValue ? 1:0;

		if (argS == "-printAccelerators") {
			p_printAccelerators=iValue;  // 0=quiet, 1=print defaults; 2=print all
		} else if (argS == "-iters") {
			p_iters = iValue;
		} else if (argS == "-mode") {
			p_hessianMode = iValue;
		} else if (argS == "-tbbThreads") {
			tbbThreads = iValue;
		} else if (argS == "-computeUnits") {
			p_computeUnits = iValue;
		} else if (argS == "-wgPerComputeUnit") {
			p_wgPerComputeUnit = iValue;
		} else if (argS == "-ampAccelerator") {
			ampAcceleratorNum = iValue;  // specify index of amp accelerator to use as default.
		} else if (argS == "-localW") {
			p_localW = iValue;  // set local shape, only for OpenCL
		} else if (argS == "-localH") {
			p_localH = iValue;  // set local shape, only for OpenCL
		} else if (argS == "-dumpIsa") {
			p_oclCompilerOpt +=" -save-temps=isa";
		} else if (argS == "-disableAvx") {
			p_oclCompilerOpt += " -fdisable-avx";
		} else if (argS == "-roiBorder") {
			p_roiBorder = iValue;  // border around picture.
		} else if (argS == "-zeroCopy") {
			p_zeroCopy = iValue;
		} else {
			printf ("error: unknown arg '%s'\n", argS.c_str());
			exit(-1);
		}
		argI++;
	};
	if (argI < argc) {
		printf ("error: all arguments must be preceded with '-'");
		exit(-1);
	}

	if (ampAcceleratorNum  != -1) {
		auto accelerators = concurrency::accelerator::get_all();
		if (ampAcceleratorNum > accelerators.size()) {
			printf ("error: requested accelerator (%d) out-of-range (max:%d)\n", ampAcceleratorNum , accelerators.size());
		};
		concurrency::accelerator::set_default(accelerators[ampAcceleratorNum].device_path);
	};

	if (p_localW * p_localH != 64) {
		printf ("warning: localW(%d) * localH (%d) != 64.  May cause inefficient code.\n", p_localW, p_localH);
	};

	if (p_printAccelerators>=2) {
		printCppAmpAccelerators();
	};

	if (p_printAccelerators>=1) {
		std::wcout << "***Default AMP Accelerator:" << std::endl;
		printAcceleratorView(concurrency::accelerator().default_view);

		std::wcout << "\n***Default OpenCL Accelerator:" << std::endl;
		printOclDevice(cl::Device::getDefault());
		std::wcout << std::endl;
	}


	tbb::task_scheduler_init init(tbbThreads ? tbbThreads : tbb::task_scheduler_init::automatic);


};


void runHessian(int argc, char* argv[]) 
{
	init(argc, argv);
	printf("info: hessianMode=%4d (0x%04x), iterations=%d\n", p_hessianMode, p_hessianMode, p_iters);
	printf("info: computeUnits=%d; wgPerComputeUnit=%d\n", p_computeUnits, p_wgPerComputeUnit);


	utils::Matrix<float> * I = BMP2matrix<float>(p_fileName, utils::Range<float>(0.0f, 255.0f));
	utils::Matrix<float>  out(I->getSize());

	long h = (long)I->getSize().height;
	long w = (long)I->getSize().width;

	//int rrr = 1;
	//Rect roi(rrr, rrr, h-rrr-1, w-rrr-1); // must be refrained
	//Rect roi( 1, 1, 319, 239 );
	//Rect roi( 1, 1, 239, 319 );

	utils::Rect rgn(0, 0, h-1, w-1);
	utils::Rect roi(p_roiBorder, p_roiBorder, h-p_roiBorder, w-p_roiBorder );

	printf ("roi: (%d,%d) (+%d,+%d) \n", 
		roi.left, roi.top, roi.width(), roi.height());
	printf ("rgn: (%d,%d) (+%d,+%d) \n", 
		rgn.left, rgn.top, rgn.width(), rgn.height());



	matrix2BMP<float>( *I, utils::Range<float>(0.0f, 255.0f), "intermediate.bmp" ); 

	H3 ht;
	ht.rot(5);
	warpFrame( out, *I, rgn, ht, rgn );
	matrix2BMP<float>( out, utils::Range<float>(0.0f, 255.0f), "warped.bmp" ); 

	H3 dH;

	int numTests = p_iters ;

	static const int bSz=512;
	char buffer[bSz]; 
	sprintf_s(buffer, bSz, " -DLOCAL_W=%d -DLOCAL_H=%d", p_localW, p_localH);
	std::string compileOpt(buffer);
	compileOpt += p_oclCompilerOpt;

	printf ("info: compile options = '%s'\n", compileOpt.c_str());
	printf ("\n");

	std::cout << "Starting tests..." << std::endl;  // flush stdout.
	if (p_hessianMode & HessianMode::e_Tbb) 
	{
		update_trz_tbb( dH,  *I,  out, 0.0f, 0.0f, roi );

		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();

			update_trz_tbb( dH,  *I,  out, 0.0f, 0.0f, roi );
		}

		const char *tag = TBB_VECTOR ? "update_trz_tbb_vector" :  "update_trz_tbb_scalar";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}

	if (p_hessianMode & HessianMode::e_TbbLamda) 
	{
		update_trz_tbb_lamda( dH,  *I,  out, 0.0f, 0.0f, roi );

		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();

			update_trz_tbb_lamda( dH,  *I,  out, 0.0f, 0.0f, roi );
		}
		const char *tag = TBB_VECTOR ? "update_trz_tbb_vector_lamda" : "update_trz_tbb_lamda";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}


	if (p_hessianMode & HessianMode::e_OclCpu) {
		MyOclContextC ocl = initOclC(e_AmdCpu);

		cl_kernel k = compileKernelC(ocl, "hessian.cl", "hessian_TRZ_x16_optbs_scalar_match2", compileOpt);

		int err;
		cl_mem I1Buf  = clCreateBuffer(ocl._context, CL_MEM_READ_ONLY, I->sizeBytes(), NULL, &err);
		CHECK_OPENCL_ERROR(err, "clCreateBuffer I1Buf");
		cl_mem outBuf = clCreateBuffer(ocl._context, CL_MEM_READ_ONLY, out.sizeBytes(), NULL, &err);
		CHECK_OPENCL_ERROR(err, "clCreateBuffer outBuf");

		clEnqueueWriteBuffer(ocl._queue, I1Buf,  true, 0, I->sizeBytes(), I->data(), 0, NULL, NULL);
		clEnqueueWriteBuffer(ocl._queue, outBuf, true, 0, out.sizeBytes(), out.data(), 0, NULL, NULL);

		update_trz_ocl(ocl, k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi);
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_ocl(ocl, k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi);
		}

		const char *tag = "update_trz_oclc_CPU_scalar_match2";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}

	if (p_hessianMode & HessianMode::e_OclGpuScalarMatch) {
		MyOclContextC ocl = initOclC(e_AmdGpu);

		cl_kernel k = compileKernelC(ocl, "hessian.cl", "hessian_TRZ_x16_optbs_scalar_match2", compileOpt);

		int err;
		cl_mem I1Buf  = clCreateBuffer(ocl._context, CL_MEM_READ_ONLY, I->sizeBytes(), NULL, &err);
		CHECK_OPENCL_ERROR(err, "clCreateBuffer I1Buf");
		cl_mem outBuf = clCreateBuffer(ocl._context, CL_MEM_READ_ONLY, out.sizeBytes(), NULL, &err);
		CHECK_OPENCL_ERROR(err, "clCreateBuffer outBuf");

		clEnqueueWriteBuffer(ocl._queue, I1Buf,  true, 0, I->sizeBytes(), I->data(), 0, NULL, NULL);
		clEnqueueWriteBuffer(ocl._queue, outBuf, true, 0, out.sizeBytes(), out.data(), 0, NULL, NULL);

		update_trz_ocl(ocl, k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi);
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_ocl(ocl, k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi);
		}

		const char *tag = "update_trz_oclc_scalar_match2";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}

	if (p_hessianMode & HessianMode::e_OclCppGpuScalarMatch) {
		cl::Kernel k = compileKernelCppDefaults("hessian.cl", "hessian_TRZ_x16_optbs_scalar_match", compileOpt);

		cl::Buffer I1Buf  = cl::Buffer(CL_MEM_READ_ONLY, I->sizeBytes());
		cl::Buffer outBuf = cl::Buffer(CL_MEM_READ_ONLY, out.sizeBytes());

		cl::enqueueWriteBuffer(I1Buf, true, 0, I->sizeBytes(), I->data());
		cl::enqueueWriteBuffer(outBuf, true, 0, out.sizeBytes(), out.data());

		update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_oclcpp_scalar_match";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};


	if (p_hessianMode & HessianMode::e_OclCppGpuScalarMatch2) {
		cl::Kernel k = compileKernelCppDefaults("hessian.cl", "hessian_TRZ_x16_optbs_scalar_match2", compileOpt);

		cl::Buffer I1Buf  = cl::Buffer(CL_MEM_READ_ONLY, I->sizeBytes());
		cl::Buffer outBuf = cl::Buffer(CL_MEM_READ_ONLY, out.sizeBytes());

		cl::enqueueWriteBuffer(I1Buf, true, 0, I->sizeBytes(), I->data());
		cl::enqueueWriteBuffer(outBuf, true, 0, out.sizeBytes(), out.data());

		update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_oclcpp_scalar_match2";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_OclCppGpuScalarMatch3) {
		cl::Kernel k = compileKernelCppDefaults("hessian.cl", "hessian_TRZ_x16_optbs_scalar_match3", compileOpt);

		cl::Buffer I1Buf  = cl::Buffer(CL_MEM_READ_ONLY, I->sizeBytes());
		cl::Buffer outBuf = cl::Buffer(CL_MEM_READ_ONLY, out.sizeBytes());

		cl::enqueueWriteBuffer(I1Buf, true, 0, I->sizeBytes(), I->data());
		cl::enqueueWriteBuffer(outBuf, true, 0, out.sizeBytes(), out.data());

		update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_oclcpp_scalar_match3";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_OclCppGpuVector) {
		cl::Kernel k = compileKernelCppDefaults("hessian.cl", "hessian_TRZ_x16_optbs_v2", compileOpt);

		cl::Buffer I1Buf  = cl::Buffer(CL_MEM_READ_ONLY, I->sizeBytes());
		cl::Buffer outBuf = cl::Buffer(CL_MEM_READ_ONLY, out.sizeBytes());

		cl::enqueueWriteBuffer(I1Buf, true, 0, I->sizeBytes(), I->data());
		cl::enqueueWriteBuffer(outBuf, true, 0, out.sizeBytes(), out.data());

		update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		}


		const char *tag = "update_trz_oclcpp_vector";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_OclCppGpuScalar) {
		cl::Kernel k = compileKernelCppDefaults("hessian.cl", "hessian_TRZ_x16_optbs_scalar", compileOpt);

		cl::Buffer I1Buf  = cl::Buffer(CL_MEM_READ_ONLY, I->sizeBytes());
		cl::Buffer outBuf = cl::Buffer(CL_MEM_READ_ONLY, out.sizeBytes());

		cl::enqueueWriteBuffer(I1Buf, true, 0, I->sizeBytes(), I->data());
		cl::enqueueWriteBuffer(outBuf, true, 0, out.sizeBytes(), out.data());

		update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_oclcpp(k, dH,  I1Buf, I->pitch(), outBuf, out.pitch(), 0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_oclcpp_scalar";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_CppAmp) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		const char *tag="?"; 
		__int64 start;

		tag = "update_trz_cppamp";
		update_trz_amp(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_amp(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		}


		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_CppAmpUseWorker) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_amp_useworker(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_amp_useworker(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		}
		const char *tag = "update_trz_cppamp_useworker";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_BoltForAmpCpu) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_boltforamp_cpu(dH,  *I, out, I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_boltforamp_cpu(dH, *I, out,  I1View, wI2View,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_boltforamp_cpu";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};


	if (p_hessianMode & HessianMode::e_BoltForAmp) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_boltforamp(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_boltforamp(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_boltforamp";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_BoltForAmpLambda) {

		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_boltforamp_lambda(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_boltforamp_lambda(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_boltforamp_lambda";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_CpuSerial) {
		update_trz_cpuserial(dH,  *I, out,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_cpuserial(dH,  *I, out,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_cpuserial";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	};

	if (p_hessianMode & HessianMode::e_BoltForAmp_Range) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_boltforamp_range(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_boltforamp_range(dH,  I1View, wI2View,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_boltforamp_range";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}

	if (p_hessianMode & HessianMode::e_BoltForAmp_Range_Cpu) {
		matrix_type  I1View(I->getSize().height, I->getSize().width, I->data());
		matrix_type  wI2View(out.getSize().height, out.getSize().width, out.data());

		update_trz_boltforamp_range_cpu(dH,  *I, out, I1View, wI2View,  0.0f, 0.0f, roi );
		__int64 start = StartProfile(); 
		for (int i=0; i<numTests; i++)
		{
			dH.zeros();
			update_trz_boltforamp_range_cpu(dH,  *I, out, I1View, wI2View,  0.0f, 0.0f, roi );
		}

		const char *tag = "update_trz_boltforamp_range_cpu";
		EndProfile(start, numTests, tag );
		printH3(dH, std::cout, tag);
	}

};


void printAllLOC() {
	printf("\n\nTotal LOC\n");
	printLOC_OclCpp();
	printLOC_AMP();
};
