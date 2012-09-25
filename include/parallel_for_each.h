#pragma once
// My quick and dirty implementation of amp.h
// To use: put a gmac_array object (not pointer or reference!)
// in your functor and use it in your kernel as if it is an array_view
// object
#include <cassert>
#include <amp.h>
#define __global __attribute__((address_space(1)))
#define SERIALIZE __cxxamp_serialize

namespace Concurrency {
static inline std::string mcw_cxxamp_fixnames(char *f) {
  std::string s(f);
  std::string out;
  for(std::string::iterator it = s.begin(); it != s.end(); it++ ) {
    if (isalnum(*it) || (*it == '_')) {
      out.append(1, *it);
    } else if (*it == '$') {
      out.append("_EC_");
    }
  }
  return out;
}

//1D sandbox_parallel_for_each
extern "C" char * kernel_source_[] asm ("_binary_kernel_cl_start");
extern "C" char * kernel_size_[] asm ("_binary_kernel_cl_size");

template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
  extent<1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  cl_int error_code;
  size_t ext = compute_domain[0];
#ifndef __GPU__
  accelerator def;
  accelerator_view accel_view = def.get_default_view();
  size_t kernel_size = (size_t)((void *)kernel_size_);
  char *kernel_source = (char*)malloc(kernel_size+1);
  memcpy(kernel_source, kernel_source_, kernel_size);
  kernel_source[kernel_size] = '\0';
  const char *ks = kernel_source;
  cl_program program =
    clCreateProgramWithSource(
      accel_view.clamp_get_context(), 1, &ks,
	NULL, &error_code);
  CHECK_ERROR(error_code, "clCreateProgramWithSource");
  error_code = clBuildProgram(program, 1, &accel_view.clamp_get_device(),
      "-D__ATTRIBUTE_WEAK__=", NULL, NULL);
  // CHECK_ERROR(error_code, "clBuildProgram");
  if (error_code != CL_SUCCESS) {
    char buf[65536] = { '\0', };
    clGetProgramBuildInfo(program,
      accel_view.clamp_get_device(), CL_PROGRAM_BUILD_LOG, 65536,
        buf, NULL);
    printf ("%s\n", buf);
    return;
  }
  //Invoke Kernel::__cxxamp_trampoline as an OpenCL kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
  std::string transformed_kernel_name =
      mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());

  std::cerr << "Kernel name = "<< transformed_kernel_name <<"\n";
  cl_kernel kernel = clCreateKernel(program,
      transformed_kernel_name.c_str(), &error_code);
  CHECK_ERROR(error_code, "clCreateKernel");
  Concurrency::Serialize s(accel_view.clamp_get_context(), kernel);
  f.SERIALIZE(s);
  error_code = clEnqueueNDRangeKernel(accel_view.clamp_get_command_queue(), kernel, 1, NULL,
      &ext, NULL, 0, NULL, NULL);
  CHECK_ERROR(error_code, "clEnqueueNDRangeKernel");
  error_code = clFinish(accel_view.clamp_get_command_queue());
  CHECK_ERROR(error_code, "clFinish");
  clReleaseKernel(kernel);
  clReleaseProgram(program);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
#endif
}
} // namespace Concurrency
