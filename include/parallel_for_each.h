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
static inline std::string mcw_cxxamp_fixnames(char *f) restrict(cpu,amp) {
    std::string str(f);
    std::string s;
    std::string out;

    std::string sE_("trampolineE_");
    std::string sE__("trampolineE__");
    size_t found_;
    size_t foundstr;
    found_ = str.find(sE_);
    foundstr = str.find(sE__);

    if(found_ != std::string::npos) {
      str.replace(int(found_)+11, 2,"");
    }
    if(foundstr != std::string::npos) {
        str.replace(int(foundstr)+11, 3,"");
    }
    s = str;

    for(std::string::iterator it = s.begin(); it != s.end(); it++ ) {
      if (isalnum(*it) || (*it == '_')) {
        out.append(1, *it);
      } else if (*it == '$') {
        out.append("_EC_");
      }
    }
    return out;
}

extern "C" char * kernel_source_[] asm ("_binary_kernel_cl_start");
extern "C" char * kernel_size_[] asm ("_binary_kernel_cl_size");
template<typename Kernel, int dim_ext>
static inline void mcw_cxxamp_launch_kernel(size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
  cl_int error_code;
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
#if 0
  std::cerr << "Kernel name = "<< transformed_kernel_name <<"\n";
#endif
  cl_kernel kernel = clCreateKernel(program,
      transformed_kernel_name.c_str(), &error_code);
  CHECK_ERROR(error_code, "clCreateKernel");
  Concurrency::Serialize s(accel_view.clamp_get_context(), kernel);
  f.SERIALIZE(s);
  error_code = clEnqueueNDRangeKernel(accel_view.clamp_get_command_queue(),
      kernel, dim_ext, NULL,
      ext, local_size, 0, NULL, NULL);
  CHECK_ERROR(error_code, "clEnqueueNDRangeKernel");
  error_code = clFinish(accel_view.clamp_get_command_queue());
  CHECK_ERROR(error_code, "clFinish");
  clReleaseKernel(kernel);
  clReleaseProgram(program);
}
template class index<1>;
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  size_t ext = compute_domain[0];
#ifndef __GPU__
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
#endif
}

template class index<2>;
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
#ifndef __GPU__
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
#endif
}

//1D parallel_for_each, tiled
template <int D0, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim0;
#ifndef __GPU__
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, &tile, f);
#else //ifndef __GPU__
  tiled_index<D0> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
#endif
}

//2D parallel_for_each, tiled
template <int D0, int D1, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0, D1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
		    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { compute_domain.tile_dim1,
                     compute_domain.tile_dim0};
#ifndef __GPU__
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, tile, f);
#else //ifndef __GPU__
  tiled_index<D0, D1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int foo = reinterpret_cast<intptr_t>(&Kernel::__cxxamp_trampoline);
#endif
}
} // namespace Concurrency
