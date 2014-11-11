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

#if !defined( BOLT_MYOCL_H )
#define BOLT_MYOCL_H

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.hpp"
#if !defined(_WIN32)
#include <unistd.h>
#endif

enum t_DeviceType  {e_Cpu, e_Gpu, e_All};

// Init OCL Platform, Context, etc.
struct MyOclContext {
	cl::Context      _context;
	cl::Device  	 _device;
	cl::CommandQueue _queue;
};

extern void printDevice(const cl::Device &d);
extern void printContext(const cl::Context &c);
extern MyOclContext initOcl(cl_int clDeviceType, int deviceIndex=0, int verbose=0) ;
extern cl::CommandQueue getQueueFromContext(cl::Context context, cl_int clDeviceType, int deviceIndex) ;
extern cl::Kernel compileKernelCpp(const MyOclContext &ocl, const char *kernelFile, const char *kernelName, std::string compileOpt);

#endif //BOLT_MYOCL_H