//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cassert>
#include <amp.h>

namespace Concurrency {
namespace CLAMP {
extern void MatchKernelNames( std::string & );
extern void CompileKernels(void);
extern void *CreateOkraKernel(std::string);
extern void OkraLaunchKernel(void *ker, size_t, size_t *global, size_t *local);
}
static inline std::string mcw_cxxamp_fixnames(char *f) restrict(cpu) {
    std::string s(f);
    std::string out;

    for(std::string::iterator it = s.begin(); it != s.end(); it++ ) {
      if (*it == '_' && it == s.begin()) {
        continue;
      } else if (isalnum(*it) || (*it == '_')) {
        out.append(1, *it);
      } else if (*it == '$') {
        out.append("_EC_");
      }
    }
    CLAMP::MatchKernelNames(out);
    return out;
}

static std::set<std::string> __mcw_cxxamp_kernels;
template<typename Kernel, int dim_ext>
static inline void mcw_cxxamp_launch_kernel(size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
#ifdef CXXAMP_ENABLE_HSA_OKRA
  //Invoke Kernel::__cxxamp_trampoline as an HSAkernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string transformed_kernel_name =
          mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
#if 0
      std::cerr << "Kernel name = "<< transformed_kernel_name <<"\n";
#endif
      kernel = CLAMP::CreateOkraKernel(transformed_kernel_name);
  }
  Concurrency::Serialize s(kernel);
  f.__cxxamp_serialize(s);
  CLAMP::OkraLaunchKernel(kernel, dim_ext, ext, local_size);
#else
  ecl_error error_code;
  accelerator def;
  accelerator_view accel_view = def.get_default_view();
  CLAMP::CompileKernels();
  //Invoke Kernel::__cxxamp_trampoline as an OpenCL kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
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

  {
    // C++ AMP specifications
    // The maximum number of tiles per dimension will be no less than 65535.
    // The maximum number of threads in a tile will be no less than 1024.
    // In 3D tiling, the maximal value of D0 will be no less than 64.
    ecl_accelerator_info accInfo;
    unsigned number = eclGetAcceleratorInfo(eclGetCurrentAcceleratorId(), &accInfo);
    #ifdef __GPU__
    assert(accInfo.maxDimensions >= dim_ext);
    assert(accInfo.maxSizes);
    #endif
    bool is = true;
    int threads_per_tile = 1;
    for(int i=0; local_size && i<dim_ext; i++) {
      threads_per_tile *= local_size[i];
      // For the following cases, set local_size=NULL and let OpenCL driver arranges it instead
      //(1) tils number exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES per dimension
      //(2) threads in a tile exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES
      //Note that the driver can still handle unregular tile_dim, e.g. tile_dim is undivisble by 2
      //So skip this condition ((local_size[i]!=1) && (local_size[i] & 1))
      if(local_size[i] > accInfo.maxSizes[i] ||threads_per_tile > accInfo.maxSizes[i]) {
        is=false;
        break;
       }
    }
    if(!is)
      local_size = NULL;
  }

  error_code = eclCallNDRange(kernel, dim_ext, NULL,
      ext, local_size);
  if (error_code != eclSuccess) {
    std::cerr << "clamp: error invoking GPU kernel;";
    std::cerr << " GMAC error code="<< error_code <<"\n";
    for (int i = 0; i<dim_ext;i++) {
      std::cerr << "global["<<i<<"] = "<<ext[i]<<"; local[";
      std::cerr << i << "] = "<<local_size[i]<<"\n";
    }
  }
#endif //CXXAMP_ENABLE_HSA_OKRA
#endif // __GPU__
}

template <int N, typename Kernel, typename _Tp>
struct pfe_helper
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
        int i;
        for (i = 0; i < k.ext[N - 1]; ++i) {
            idx[N - 1] = i;
            pfe_helper<N - 1, Kernel, _Tp>::call(k, idx);
        }
    }
};
template <typename Kernel, typename _Tp>
struct pfe_helper<0, Kernel, _Tp>
{
    static inline void call(Kernel& k, _Tp& idx) restrict(amp,cpu) {
        k.k(idx);
    }
};

template <int N, typename Kernel>
class pfe_wrapper
{
public:
    explicit pfe_wrapper(extent<N>& other, const Kernel& f) restrict(amp,cpu)
        : ext(other), k(f) {}
    void operator() (index<N> idx) restrict(amp,cpu) {
        pfe_helper<N - 3, pfe_wrapper<N, Kernel>, index<N>>::call(*this, idx);
    }
private:
    const extent<N> ext;
    const Kernel k;
    template <int K, typename Ker, typename _Tp>
        friend struct pfe_helper;
};

template <int N, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<N> compute_domain, const Kernel& f) restrict(cpu, amp) {
#ifndef __GPU__
    for(int i = 0 ; i < N ; i++)
    {
      if(compute_domain[i]<=0)
        throw invalid_compute_domain("Extent is less or equal than 0.");
    }

    size_t ext[3] = {static_cast<size_t>(compute_domain[N - 1]),
        static_cast<size_t>(compute_domain[N - 2]),
        static_cast<size_t>(compute_domain[N - 3])};
    const pfe_wrapper<N, Kernel> _pf(compute_domain, f);
    mcw_cxxamp_launch_kernel<pfe_wrapper<N, Kernel>, 3>(ext, NULL, _pf);
#else
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif
}

template class index<1>;
//1D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext = compute_domain[0];
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

template class index<2>;
//2D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

template class index<3>;
//3D parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    extent<3> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
                   static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  mcw_cxxamp_launch_kernel<Kernel, 3>(ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

//1D parallel_for_each, tiled
template <int D0, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim0;
  static_assert( compute_domain.tile_dim0 <= 1024, "The maximum nuimber of threads in a tile is 1024");
  if(ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, &tile, f);
#else //ifndef __GPU__
  tiled_index<D0> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

//2D parallel_for_each, tiled
template <int D0, int D1, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0, D1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { compute_domain.tile_dim1,
                     compute_domain.tile_dim0};
  static_assert( (compute_domain.tile_dim1 * compute_domain.tile_dim0)<= 1024, "The maximum nuimber of threads in a tile is 1024");
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, tile, f);
#else //ifndef __GPU__
  tiled_index<D0, D1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
 //3D parallel_for_each, tiled
template <int D0, int D1, int D2, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0, D1, D2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  size_t ext[3] = { static_cast<size_t>(compute_domain[2]),
                    static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[3] = { compute_domain.tile_dim2,
                     compute_domain.tile_dim1,
                     compute_domain.tile_dim0};
  static_assert(( compute_domain.tile_dim2 * compute_domain.tile_dim1* compute_domain.tile_dim0)<= 1024, "The maximum nuimber of threads in a tile is 1024");
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0) || (ext[2] % tile[2] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  mcw_cxxamp_launch_kernel<Kernel, 3>(ext, tile, f);
#else //ifndef __GPU__
  tiled_index<D0, D1, D2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
} // namespace Concurrency
