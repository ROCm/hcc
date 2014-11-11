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

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>

#include "myocl_bufferpool.h"

#include "hessian_oclcpp.h"

#include "matrix_utils.h"
#include "hessian.h"

OclBufferPool bufferPool;

int g_oclcpp_launch_end;
int g_oclcpp_launch_start = __LINE__+1;
bool update_trz_oclcpp(cl::Kernel k, H3& dH, const cl::Buffer& I1, int ipitch, const cl::Buffer& wI2, int wpitch, float sigma, float gradThresh, const  utils::Rect& roi )
{
	// Compute number of WG required to fill the machine:
	const int wgPerComputeUnit=p_wgPerComputeUnit; // pick a number high enough to hide latency
	const int computeUnits = p_computeUnits ? p_computeUnits : cl::Device::getDefault().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	int elements     = wgPerComputeUnit * computeUnits * 16; 
	int sizeBytes = elements * sizeof(float);

	int localW = p_localW;
	int localH = p_localH;

	int globalW = computeUnits*localW;
	int globalH = wgPerComputeUnit*localH;
	globalW = (roi.width()  < globalW) ? roi.width() : globalW; // FIXME - makes global size not a multiple of local size.
	globalH = (roi.height() < globalH) ? roi.height() : globalH;

	cl::Buffer dest = bufferPool.alloc(p_zeroCopy? (CL_MEM_ALLOC_HOST_PTR|CL_MEM_WRITE_ONLY) : (CL_MEM_WRITE_ONLY), sizeBytes);

	int arg=0;
	k.setArg(arg++, dest);
	k.setArg(arg++, I1); 
	k.setArg(arg++, ipitch); 
	k.setArg(arg++, wI2);
	k.setArg(arg++, wpitch); 

	k.setArg(arg++, roi.top+1);
	k.setArg(arg++, roi.left+1); 
	k.setArg(arg++, roi.bottom-1);
	k.setArg(arg++, roi.right-1); 

	float Xc = (roi.left + roi.right) / 2.0f;
	float Yc = (roi.top + roi.bottom) / 2.0f;

	k.setArg(arg++, Xc);
	k.setArg(arg++, Yc); 

	float sigma_sq = sigma * sigma;
	k.setArg(arg++, gradThresh);  //thres
	k.setArg(arg++, sigma_sq);  //siqma_sq

	cl::CommandQueue::getDefault().enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(globalW,globalH), cl::NDRange(localW, localH));
	g_oclcpp_launch_end = __LINE__-1;
	// Map the data back and run the computation on the CPU:
	float * destH = (float*)cl::CommandQueue::getDefault().enqueueMapBuffer(dest, true, CL_MAP_READ, 0/*offset*/, sizeBytes);
	bufferPool.free(dest);

	{
		//sumBlocks(hessian, (float*)mappedDst, elements);
		const int NH = 16;
		float hessian[NH];
		for (int i=0; i<NH; i++) hessian[i] = 0.0f;
		for (int i=0; i<elements; i++)
			hessian[i%NH] += destH[i];

		double H[4][4];
		double b[4];

		for (int i=0,k=0; i<5; i++)
		{
			if (i == 4)
			{
				for (int j=0; j<4; j++) 
					b[j] = (double)hessian[k++];
			}
			else
			{
				for (int j=0; j<=i; j++) 
					H[i][j] = H[j][i] = (double)hessian[k++];
			}
		}


		double dm[4];
		dH.zeros();

		if (solve_ChD<double,4> (dm, H, b))
		{
			dH[0] = dH[4] = dm[0];
			dH[1] = dm[1];
			dH[2] = dm[2];
			dH[5] = dm[3];
			dH[3] = -dH[1];
			return true;
		}

		return false;
	}

	return true;
}


void printLOC_OclCpp() 
{
	const char *tag = "oclcpp";
	printf("\n");
	printf("%s launch\t:\t%d\n", tag, g_oclcpp_launch_end - g_oclcpp_launch_start);
};
