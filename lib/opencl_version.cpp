#include <iostream>
#include <string>
#include <cassert>
#include <CL/cl.h>

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

