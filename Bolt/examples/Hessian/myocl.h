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

#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "CL/cl.hpp"


// Init OCL Platform, Context, etc.  C++ version.
struct MyOclContextCpp {
	cl::Context      _context;
	cl::Device  	 _device;
	cl::CommandQueue _queue;
};

// A C-version with key information.
struct MyOclContextC {
	cl_context      _context;
	cl_device_id  	 _device;
	cl_command_queue _queue;
};

enum t_DeviceType  {e_IntelCpu, e_AmdCpu, e_AmdGpu, e_All};

// Utility functions:
extern void  CHECK_OPENCL_ERROR(cl_int err, const char * name);


// CPP versions:
extern  MyOclContextCpp initOclCpp(t_DeviceType useGpu);
extern cl::Kernel compileKernelCpp(const MyOclContextCpp &ocl, const char *kernelFile, const char *kernelName, std::string compileOpt);

// CPP version using defaults:
extern cl::Kernel compileKernelCppDefaults(const char *kernelFile, const char *kernelName, std::string compileOpt);

// C versions
extern  MyOclContextC initOclC(t_DeviceType useGpu);
extern cl_kernel compileKernelC(const MyOclContextC &ocl, const char *kernelFile, const char *kernelName, std::string compileOpt);

