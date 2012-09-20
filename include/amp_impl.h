#ifndef INCLUDE_AMP_IMPL_H
#define INCLUDE_AMP_IMPL_H

#include <iostream>
#include <CL/cl.h>

#define CHECK_ERROR(error_code, message) \
  if (error_code != CL_SUCCESS) { \
    std::cout << "Error: " << message << "\n"; \
    std::cout << "Code: " << error_code << "\n"; \
    std::cout << "Line: " << __LINE__ << "\n"; \
    exit(1); \
  }

// Specialization of AMP classes/templates

namespace Concurrency {
extern "C" int get_global_id(int n) restrict(amp);
//Accelerators
accelerator::accelerator(void) {
  cl_int error_code;
  error_code = clGetPlatformIDs(1, &platform_, NULL);
  CHECK_ERROR(error_code, "clGetPlatformIDs");
  error_code = clGetDeviceIDs(platform_,
    CL_DEVICE_TYPE_GPU, 1, &device_, NULL);
  CHECK_ERROR(error_code, "clGetDeviceIDs");
}
accelerator_view accelerator::create_view(void) {
  accelerator_view sa(device_);
  return sa;
}

accelerator_view::accelerator_view(const accelerator_view &v):
  device_(v.device_), context_(v.context_),
  command_queue_(v.command_queue_) {
  //std::cerr << "SAV: copy constructor calld. This = " << this;
  //std::cerr << " v = " << &v << std::endl;
  cl_int err = clRetainCommandQueue(command_queue_);
  CHECK_ERROR(err, "clRetainCommandQueue"); 
}

accelerator_view::~accelerator_view() {
  clReleaseCommandQueue(command_queue_);
  clReleaseContext(context_);
}

accelerator_view::accelerator_view(cl_device_id d):
    device_(d) {
  cl_int error_code;
  context_ = clCreateContext(0, 1, &d, NULL, NULL, &error_code);
  CHECK_ERROR(error_code, "clCreateContext");
  command_queue_ = clCreateCommandQueue(context_, d, 0, &error_code);
  CHECK_ERROR(error_code, "clCreateCommandQueue");
}
} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
