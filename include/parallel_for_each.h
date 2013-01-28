#pragma once
// My quick and dirty implementation of amp.h
// To use: put a gmac_array object (not pointer or reference!)
// in your functor and use it in your kernel as if it is an array_view
// object
#include <cassert>
#include <amp.h>
#define __global __attribute__((address_space(1)))

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
static bool __mcw_cxxamp_compiled = false;
static std::set<std::string> __mcw_cxxamp_kernels;
extern "C" char * kernel_source_[] asm ("_binary_kernel_cl_start");
extern "C" char * kernel_size_[] asm ("_binary_kernel_cl_size");
template<typename Kernel, int dim_ext>
static inline void mcw_cxxamp_launch_kernel(size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
  ecl_error error_code;
  accelerator def;
  accelerator_view accel_view = def.get_default_view();

  if ( !__mcw_cxxamp_compiled ) {
    size_t kernel_size = (size_t)((void *)kernel_size_);
    char *kernel_source = (char*)malloc(kernel_size+1);
    memcpy(kernel_source, kernel_source_, kernel_size);
    kernel_source[kernel_size] = '\0';
    const char *ks = kernel_source;
    error_code = eclCompileSource(ks, "-D__ATTRIBUTE_WEAK__=");
    CHECK_ERROR_GMAC(error_code, "eclCompileSource");
    __mcw_cxxamp_compiled = true;
    free(kernel_source);
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
  ecl_kernel kernel;
  auto it = __mcw_cxxamp_kernels.insert(transformed_kernel_name);
  error_code = eclGetKernel(it.first->c_str(), &kernel);
  CHECK_ERROR_GMAC(error_code, "eclGetKernel");
  Concurrency::Serialize s(kernel);
  f.__cxxamp_serialize(s);
  error_code = eclCallNDRange(kernel, dim_ext, NULL,
      ext, local_size);
  CHECK_ERROR_GMAC(error_code, "eclCallNDRange");
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

template class index<3>;
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<3> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
                   static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
#ifndef __GPU__
  mcw_cxxamp_launch_kernel<Kernel, 3>(ext, NULL, f);
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
