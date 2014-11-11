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
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <tchar.h>


#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "CL/cl.hpp"

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

void printDevice(int deviceCnt, const cl::Device &d) {

	std::cout << "Device # " << deviceCnt << ": " 
		<< d.getInfo<CL_DEVICE_NAME>()
		<< " CU="<< d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() 
		<< " Freq="<< d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "Mhz"
		<< "\n";
};


void printDeviceID(int deviceCnt, cl_device_id d) 
{
	cl_int status;
    char deviceName[1024];
    cl_uint maxComputeUnits;
    cl_uint maxClockFrequency;

    status = clGetDeviceInfo( d, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL); 
    status = clGetDeviceInfo( d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
    status = clGetDeviceInfo( d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFrequency, NULL);

		std::cout << "Device # " << deviceCnt << ": " 
		<< deviceName
		<< " CU="<< maxComputeUnits 
		<< " Freq="<< maxClockFrequency 
		<< "\n";
};


//---
// CPP versions:
MyOclContextCpp initOclCpp(t_DeviceType x_deviceType) 
{
	MyOclContextCpp ocl;

	cl_int err;
	std::vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);
	CHECK_OPENCL_ERROR(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
	std::cerr << "info: Found: " << platformList.size() << " platforms." << std::endl;

	//FIXME - add support for multiple vendors here, right now just pick the first platform.
	std::string platformVendor;
	platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	std::cout << "info: platform is: " << platformVendor << "\n";
	cl_context_properties cprops[3] = 
	{CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};

	int clDeviceType;
	if (x_deviceType == e_IntelCpu || x_deviceType == e_AmdCpu) {
		clDeviceType = CL_DEVICE_TYPE_CPU;
	} else if (x_deviceType == e_AmdGpu) {
		clDeviceType = CL_DEVICE_TYPE_GPU;
	} else if (x_deviceType == e_All) {
		clDeviceType = CL_DEVICE_TYPE_ALL;
	} else {
		CHECK_OPENCL_ERROR(-1, "Bad requested device type");
	}

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

	int deviceCnt = 0;
	std::for_each(devices.begin(), devices.end(), [&] (cl::Device &d) {
		printDevice(deviceCnt, d);
		deviceCnt ++;
	});

	int deviceId = 0; // pick first device.
	std::cout << "info: selected device #" << deviceId;
	printDevice(deviceId, devices[deviceId]);
	ocl._device = devices[deviceId];

	cl::CommandQueue q(ocl._context, ocl._device);
	ocl._queue = q;

	return ocl;
};


cl::Kernel compileKernelCpp(const MyOclContextCpp &ocl, const char *kernelFile, const char *kernelName, std::string compileOpt)
{
	cl::Program program;
	try 
	{
		std::ifstream infile(kernelFile);
		if (infile.fail()) {
			TCHAR cCurrentPath[FILENAME_MAX];
			if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
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
	}
};

cl::Kernel compileKernelCppDefaults(const char *kernelFile, const char *kernelName, std::string compileOpt)
{
	cl::Program program;
	try 	{
		std::ifstream infile(kernelFile);
		if (infile.fail()) {
			TCHAR cCurrentPath[FILENAME_MAX];
			if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
				std::wcout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
			};
			std::cout << "ERROR: can't open file '" << kernelFile << std::endl;
			throw;
		};
		std::string str((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());

		program = cl::Program(str, false);
		compileOpt += " -x clc++";
		program.build(compileOpt.c_str());

		cl::Kernel kernel(program, kernelName);
		return kernel;
	} catch (cl::Error err) {
		std::cerr << "ERROR: "<< err.what()<< "("<< err.err()<< ")"<< std::endl;
		std::cout << "Error building program\n";
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
	}
};


int g_oclc_init_start = __LINE__+1;
MyOclContextC initOclC(t_DeviceType x_deviceType) 
{
	MyOclContextC ocl;

	cl_int status;

	cl_uint numPlatforms;
	cl_platform_id platform = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	CHECK_OPENCL_ERROR(status, "clGetPlatformIDs failed.");

	if (numPlatforms > 0) 	{
		cl_platform_id* platforms = new cl_platform_id[numPlatforms];
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		CHECK_OPENCL_ERROR(status, "clGetPlatformIDs failed.");

		std::cerr << "info: Found: " << numPlatforms << " platforms." << std::endl;

		char platformName[100];
		for (unsigned i = 0; i < numPlatforms; ++i) 
		{
			status = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_VENDOR,
				sizeof(platformName),
				platformName,
				NULL);
			CHECK_OPENCL_ERROR(status, "clGetPlatformInfo failed.");

			platform = platforms[i];
			if (!strcmp(platformName, "Advanced Micro Devices, Inc.")) 
				break;
		}
		std::cout << "Platform found : " << platformName << "\n";
		delete[] platforms;
	}

	if(NULL == platform) {
		std::cout << "NULL platform found so Exiting Application.";
		throw;
	}

	cl_context_properties cps[3] = 
	{CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};

	int clDeviceType;
	if (x_deviceType == e_IntelCpu || x_deviceType == e_AmdCpu) {
		clDeviceType = CL_DEVICE_TYPE_CPU;
	} else if (x_deviceType == e_AmdGpu) {
		clDeviceType = CL_DEVICE_TYPE_GPU;
	} else if (x_deviceType == e_All) {
		clDeviceType = CL_DEVICE_TYPE_ALL;
	} else {
		CHECK_OPENCL_ERROR(-1, "Bad requested device type");
	}

	ocl._context = clCreateContextFromType(cps, clDeviceType, NULL,
		NULL,
		&status);
	CHECK_OPENCL_ERROR(status, "Context::Context()");  

	// Find devices:
	size_t sizeDevicesInBytes;
	status = clGetContextInfo( ocl._context, CL_CONTEXT_DEVICES, 0, NULL, &sizeDevicesInBytes);
	CHECK_OPENCL_ERROR(status, "clGetContextInfo(CL_CONTEXT_DEVICES) failed.");

	cl_device_id *devices = (cl_device_id *)malloc( sizeDevicesInBytes );
	/* grab the handles to all of the devices in the program. */
	status = clGetContextInfo( ocl._context, CL_CONTEXT_DEVICES, sizeDevicesInBytes, devices, NULL);
	CHECK_OPENCL_ERROR(status, "clGetContextInfo(CL_CONTEXT_DEVICES) failed.");


	size_t numDevices = sizeDevicesInBytes/sizeof(cl_device_id);
	int deviceCnt = 0;
	std::for_each(devices, devices+numDevices, [&] (cl_device_id &d) {
		printDevice(deviceCnt, d);
		deviceCnt ++;
	});

	int deviceId = 0; // pick first device.
	std::cout << "info: selected ";
	printDevice(deviceId, devices[deviceId]);
	ocl._device = devices[deviceId];

	ocl._queue = clCreateCommandQueue(ocl._context, ocl._device, 0, &status);
	CHECK_OPENCL_ERROR(status, "clCreateCommandQueue failed.");

	if(devices != NULL) {
		free(devices);
		devices = NULL;
	}

	return ocl;
};
int g_oclc_init_end = __LINE__-1;


cl_kernel compileKernelC(const MyOclContextC &ocl, const char *kernelFile, const char *kernelName,std::string compileOpt)
{
	cl_int status = CL_SUCCESS;

	std::ifstream infile(kernelFile);
	if (infile.fail()) {
		TCHAR cCurrentPath[FILENAME_MAX];
		if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
			std::wcout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
		};
		std::cout << "ERROR: can't open file '" << kernelFile << std::endl;
		throw;
	};
	std::string sourceStr((std::istreambuf_iterator<char>(infile)), std::istreambuf_iterator<char>());
	const char* kStr = sourceStr.c_str();
	size_t sourceSize[] = {strlen(kStr)};

	cl_program program = clCreateProgramWithSource( ocl._context, 1, &kStr, sourceSize, &status);
	CHECK_OPENCL_ERROR(status, "clCreateProgramWithSource failed.");

	std::string buildOptions = "-x clc++ " + compileOpt;  // Support C++ in the rest of the file.
	status = clBuildProgram( program, 0, NULL, buildOptions.c_str(), NULL, NULL);
	if (status != CL_SUCCESS) {
		if(status == CL_BUILD_PROGRAM_FAILURE)
		{
			cl_int logStatus;
			char *buildLog = NULL;
			size_t buildLogSize = 0;
			logStatus = clGetProgramBuildInfo ( program, ocl._device, CL_PROGRAM_BUILD_LOG, 
                                                buildLogSize, buildLog, &buildLogSize);
			CHECK_OPENCL_ERROR(logStatus, "clGetProgramBuildInfo failed.");

			buildLog = (char*)malloc(buildLogSize);
			memset(buildLog, 0, buildLogSize);

			logStatus = clGetProgramBuildInfo ( program, ocl._device, CL_PROGRAM_BUILD_LOG, 
                                                buildLogSize, buildLog, NULL);
			CHECK_OPENCL_ERROR(logStatus, "clGetProgramBuildInfo failed.");

			std::cout << " \n\t\t\tBUILD LOG\n";
			std::cout << " ************************************************\n";
			std::cout << buildLog << std::endl;
			std::cout << " ************************************************\n";
			free(buildLog);
		}
		std::cout << "Error building program\n";
	}
	CHECK_OPENCL_ERROR(status, "clBuildProgram failed.");

	cl_kernel k = clCreateKernel(program, kernelName, &status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");
	return k;
};
