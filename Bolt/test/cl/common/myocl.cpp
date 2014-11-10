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

#include <iostream>
#include <fstream>
#include <algorithm>

#include <bolt/unicode.h>
#include "myocl.h"


void  CHECK_OPENCL_ERROR(cl_int err, const char * name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name
			<< " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void printDevice(const cl::Device &d) {

	std::cout 
		<< d.getInfo<CL_DEVICE_NAME>()
		<< " CU="<< d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() 
		<< " Freq="<< d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "Mhz"
		<< "\n";
};

void printContext(const cl::Context &c) 
{
	cl_int numDevices = c.getInfo<CL_CONTEXT_NUM_DEVICES>();

	std::cout <<"Context: #devices=" << numDevices
		//<< ", props=" << c.getInfo<CL_CONTEXT_PROPERTIES>() 
		<< ", refCnt=" << c.getInfo<CL_CONTEXT_REFERENCE_COUNT>()
		<< "\n";

	std::vector<cl::Device> devices;
	devices = c.getInfo<CL_CONTEXT_DEVICES>();

	int deviceCnt = 0;
	std::for_each(devices.begin(), devices.end(), [&] (cl::Device &d) {
		std::cout << "#" << deviceCnt << "  ";
		printDevice(d);
		deviceCnt ++;
	}); 
};



MyOclContext initOcl(cl_int clDeviceType, int deviceIndex, int verbose) 
{
	MyOclContext ocl;

	cl_int err;
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	CHECK_OPENCL_ERROR(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
	if (verbose) {
		std::cerr << "info: Found: " << platformList.size() << " platforms." << std::endl;
	}

	//FIXME - add support for multiple vendors here, right now just pick the first platform.
	std::string platformVendor;
	platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	if (verbose) {
		std::cout << "info: platform is: " << platformVendor << "\n";
	}
	cl_context_properties cprops[3] = 
	{CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};



	cl::Context context(
		clDeviceType, 
		cprops,
		NULL,
		NULL,
		&err);
	CHECK_OPENCL_ERROR(err, "Context::Context()");  
	ocl._context = context;


	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	CHECK_OPENCL_ERROR(
		devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");


	if (0) {
		int deviceCnt = 0;
		std::for_each(devices.begin(), devices.end(), [&] (cl::Device &d) {
			std::cout << "#" << deviceCnt << "  ";
			printDevice(d);
			deviceCnt ++;
		});
	}

	if (verbose) {
		std::cout << "info: selected device #" << deviceIndex << "  ";
		printDevice(devices[deviceIndex]);
	}
	ocl._device = devices[deviceIndex];

	cl::CommandQueue q(ocl._context, ocl._device);
	ocl._queue = q;

	return ocl;
};


cl::CommandQueue getQueueFromContext(cl::Context context, cl_int clDeviceType, int deviceIndex) 
{
	std::vector< cl::Device > devices = context.getInfo< CL_CONTEXT_DEVICES >();
	for (std::vector<cl::Device>::iterator iter=devices.begin(); iter != devices.end(); iter++) {
		if (iter->getInfo<CL_DEVICE_TYPE>() == clDeviceType) {
			if (--deviceIndex < 0) {
				::cl::CommandQueue myQueue( context, *iter );
				return myQueue;
			}
		}
	}

	throw;  // no suitable device found.
}



cl::Kernel compileKernelCpp(const MyOclContext &ocl, const char *kernelFile, const char *kernelName, std::string compileOpt)
{
	cl::Program program;
	try 
	{
		std::ifstream infile(kernelFile);
		if (infile.fail()) {
			TCHAR cCurrentPath[FILENAME_MAX];
#if defined(_WIN32)
			if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR)))
#else
			if (getcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) 
#endif
			{
				std::wcout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
			};
			std::cout << "ERROR: can't open file '" << kernelFile << std::endl;
			throw;
		};

		std::string str((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

		const char* kStr = str.c_str();
		cl::Program::Sources source(1, std::make_pair(kStr,strlen(kStr)));

		program = cl::Program(ocl._context, source);
		std::vector<cl::Device> devices;
		devices.push_back(ocl._device);
		compileOpt += " -x clc++";
		program.build(devices, compileOpt.c_str());

		cl::Kernel kernel(program, kernelName);

		return kernel;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: "<< err.what()<< "("<< err.err()<< ")"<< std::endl;
		std::cout << "Error building program\n";
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(ocl._device) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(ocl._device) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ocl._device) << std::endl;
		return cl::Kernel();
	}
};


