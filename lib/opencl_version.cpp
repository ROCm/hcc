#include <iostream>
#include <string>
#include <cassert>
#include <CL/cl.h>

//
// get OpenCL version
//
// returns 11 for OpenCL 1.1, 12 for OpenCL 1.2, 20 for OpenCL 2.0
//
extern "C" int GetOpenCLVersion() {
  cl_uint num_platforms = 0;
  cl_int err;
  cl_platform_id platform_id[1];
  char platform_version[256];
  size_t platform_version_length;
  int i;
  err = clGetPlatformIDs(1, platform_id, &num_platforms);
  assert(num_platforms > 0);
  err = clGetPlatformInfo(platform_id[0], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, &platform_version_length);
  assert(platform_version_length > 0);

  // OpenCL version string. Returns the OpenCL version supported by the implementation. This version string has the following format:
  // OpenCL<space><major_version.minor_version><space><platform-specific information>
  std::string str(platform_version);
  std::string::size_type pos1 = str.find(' ');
  std::string::size_type pos2 = str.find('.', pos1 + 1);
  std::string::size_type pos3 = str.find(' ', pos2 + 1);

  // OpenCL 1.1 will have return value 11
  // OpenCL 1.2 will have return value 12
  // OpenCL 2.0 will have return value 20
  return std::stoi(str.substr(pos1 + 1, pos2 - pos1 - 1)) * 10 +
         std::stoi(str.substr(pos2 + 1, pos3 - pos2 - 1));
}

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

