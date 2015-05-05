#include <iostream>
#include <string>
#include <cassert>
#include <CL/cl.h>

//
// Utility function to test if a certain CL extension is available
//
static int IsCLExtensionAvailable(const char* extension_name) {
  cl_uint num_platforms = 0;
  cl_int err;
  cl_platform_id platform_id[1];
  err = clGetPlatformIDs(1, platform_id, &num_platforms);
  assert(num_platforms > 0);
  
  cl_uint num_devices = 0;
  cl_device_id device_id[1];
  // try GPU first
  err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 1, device_id, &num_devices);
  if (err != CL_SUCCESS) {
    // try CPU then
    err = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_CPU, 1, device_id, &num_devices);
    if (err != CL_SUCCESS) {
      // can't find any OpenCL device
      return 0;
    }
  }

  assert(num_devices > 0);

  size_t size_extensions = 0;
  // use clGetDeviceInfo to check if certain extension is available
  err = clGetDeviceInfo(device_id[0], CL_DEVICE_EXTENSIONS, 0, NULL, &size_extensions);
  if (err != CL_SUCCESS) {
    // can't get device info
    return 0;
  }

  assert(size_extensions > 0);
  char* extensions = new char[size_extensions + 1];
  err = clGetDeviceInfo(device_id[0], CL_DEVICE_EXTENSIONS, size_extensions, extensions, NULL);
  if (err != CL_SUCCESS) {
    // can't get device info
    delete[] extensions;
    return 0;
  }
  extensions[size_extensions] = '\0';

  //std::cout << extensions << std::endl;

  if (std::string(extensions).find(extension_name) != std::string::npos) {
    // found the extension
    delete[] extensions;
    return 1;
  } 
  delete[] extensions;
  return 0;
}

//
// check if SPIR is available on the system
//
extern "C" int IsSPIRAvailable() {
  return IsCLExtensionAvailable("cl_khr_spir");
}

