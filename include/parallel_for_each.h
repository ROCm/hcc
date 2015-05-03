//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cassert>
#include <future>
#include <utility>
#ifdef __AMP_CPU__
#include <thread>
#endif

#include <amp.h>
#include <amp_runtime.h>

namespace Concurrency {

#ifdef __AMP_CPU__
#define SSIZE 1024 * 10
const unsigned int NTHREAD = std::thread::hardware_concurrency();
template <int N, typename Kernel,  int K>
struct cpu_helper
{
    static inline void call(const Kernel& k, index<K>& idx, const extent<K>& ext) restrict(amp,cpu) {
        int i;
        for (i = 0; i < ext[N]; ++i) {
            idx[N] = i;
            cpu_helper<N + 1, Kernel, K>::call(k, idx, ext);
        }
    }
};
template <typename Kernel, int K>
struct cpu_helper<K, Kernel, K>
{
    static inline void call(const Kernel& k, const index<K>& idx, const extent<K>& ext) restrict(amp,cpu) {
        (const_cast<Kernel&>(k))(idx);
    }
};

template <typename Kernel, int N>
void partitioned_task(const Kernel& ker, const extent<N>& ext, int part) {
    index<N> idx;
    int start = ext[0] * part / NTHREAD;
    int end = ext[0] * (part + 1) / NTHREAD;
    for (int i = start; i < end; i++) {
        idx[0] = i;
        cpu_helper<1, Kernel, N>::call(ker, idx, ext);
    }
}

template <typename Kernel, int D0>
void partitioned_task_tile(Kernel const& f, tiled_extent<D0> const& ext, int part) {
    int start = (ext[0] / D0) * part / NTHREAD;
    int end = (ext[0] / D0) * (part + 1) / NTHREAD;
    int stride = end - start;
    if (stride == 0)
        return;
    char *stk = new char[D0 * SSIZE];
    tiled_index<D0> *tidx = new tiled_index<D0>[D0];
    tile_barrier::pb_t amp_bar = std::make_shared<barrier_t>(D0);
    tile_barrier tbar(amp_bar);
    for (int tx = start; tx < end; tx++) {
        int id = 0;
        char *sp = stk;
        tiled_index<D0> *tip = tidx;
        for (int x = 0; x < D0; x++) {
            new (tip) tiled_index<D0>(tx * D0 + x, x, tx, tbar);
            amp_bar->setctx(++id, sp, f, tip, SSIZE);
            sp += SSIZE;
            ++tip;
        }
        amp_bar->idx = 0;
        while (amp_bar->idx == 0) {
            amp_bar->idx = id;
            amp_bar->swap(0, id);
        }
    }
    delete [] stk;
    delete [] tidx;
}
template <typename Kernel, int D0, int D1>
void partitioned_task_tile(Kernel const& f, tiled_extent<D0, D1> const& ext, int part) {
    int start = (ext[0] / D0) * part / NTHREAD;
    int end = (ext[0] / D0) * (part + 1) / NTHREAD;
    int stride = end - start;
    if (stride == 0)
        return;
    char *stk = new char[D1 * D0 * SSIZE];
    tiled_index<D0, D1> *tidx = new tiled_index<D0, D1>[D0 * D1];
    tile_barrier::pb_t amp_bar = std::make_shared<barrier_t>(D0 * D1);
    tile_barrier tbar(amp_bar);

    for (int tx = 0; tx < ext[1] / D1; tx++)
        for (int ty = start; ty < end; ty++) {
            int id = 0;
            char *sp = stk;
            tiled_index<D0, D1> *tip = tidx;
            for (int x = 0; x < D1; x++)
                for (int y = 0; y < D0; y++) {
                    new (tip) tiled_index<D0, D1>(D1 * tx + x, D0 * ty + y, x, y, tx, ty, tbar);
                    amp_bar->setctx(++id, sp, f, tip, SSIZE);
                    ++tip;
                    sp += SSIZE;
                }
            amp_bar->idx = 0;
            while (amp_bar->idx == 0) {
                amp_bar->idx = id;
                amp_bar->swap(0, id);
            }
        }
    delete [] stk;
    delete [] tidx;
}

template <typename Kernel, int D0, int D1, int D2>
void partitioned_task_tile(Kernel const& f, tiled_extent<D0, D1, D2> const& ext, int part) {
    int start = (ext[0] / D0) * part / NTHREAD;
    int end = (ext[0] / D0) * (part + 1) / NTHREAD;
    int stride = end - start;
    if (stride == 0)
        return;
    char *stk = new char[D2 * D1 * D0 * SSIZE];
    tiled_index<D0, D1, D2> *tidx = new tiled_index<D0, D1, D2>[D0 * D1 * D2];
    tile_barrier::pb_t amp_bar = std::make_shared<barrier_t>(D0 * D1 * D2);
    tile_barrier tbar(amp_bar);

    for (int i = 0; i < ext[2] / D2; i++)
        for (int j = 0; j < ext[1] / D1; j++)
            for(int k = start; k < end; k++) {
                int id = 0;
                char *sp = stk;
                tiled_index<D0, D1, D2> *tip = tidx;
                for (int x = 0; x < D2; x++)
                    for (int y = 0; y < D1; y++)
                        for (int z = 0; z < D0; z++) {
                            new (tip) tiled_index<D0, D1, D2>(D2 * i + x,
                                                              D1 * j + y,
                                                              D0 * k + z,
                                                              x, y, z, i, j, k, tbar);
                            amp_bar->setctx(++id, sp, f, tip, SSIZE);
                            ++tip;
                            sp += SSIZE;
                        }
                amp_bar->idx = 0;
                while (amp_bar->idx == 0) {
                    amp_bar->idx = id;
                    amp_bar->swap(0, id);
                }
            }
    delete [] stk;
    delete [] tidx;
}
template <typename Kernel, int N>
void launch_cpu_task(extent<N> const& compute_domain, Kernel const& f)
{
    Concurrency::Serialize s(nullptr);
    f.__cxxamp_serialize(s);
    CLAMP::enter_kernel();
    std::vector<std::thread> th(NTHREAD);
    for (int i = 0; i < NTHREAD; ++i)
        th[i] = std::thread(partitioned_task<Kernel, N>, std::cref(f), std::cref(compute_domain), i);
    for (auto& t : th)
        if (t.joinable())
            t.join();
    f.__cxxamp_serialize(s);
    CLAMP::leave_kernel();
}
#endif

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
static inline std::shared_future<void>* mcw_cxxamp_launch_kernel_async(size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string transformed_kernel_name =
          mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(transformed_kernel_name);
  }
  Concurrency::Serialize s(kernel);
  f.__cxxamp_serialize(s);
  return CLAMP::LaunchKernelAsync(kernel, dim_ext, ext, local_size);
#endif
}

template<typename Kernel, int dim_ext>
static inline void mcw_cxxamp_launch_kernel(size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string transformed_kernel_name =
          mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(transformed_kernel_name);
  }
  Concurrency::Serialize s(kernel);
  f.__cxxamp_serialize(s);
  CLAMP::LaunchKernel(kernel, dim_ext, ext, local_size);
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
#ifdef __GPU__
        k.k(idx);
#endif
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
#ifdef __AMP_CPU__
    int* foo1 = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif

    size_t compute_domain_size = 1;
    for(int i = 0 ; i < N ; i++)
    {
      if(compute_domain[i]<=0)
        throw invalid_compute_domain("Extent is less or equal than 0.");
      if (static_cast<size_t>(compute_domain[i]) > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
      compute_domain_size *= static_cast<size_t>(compute_domain[i]);
      if (compute_domain_size > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
    }

    size_t ext[3] = {static_cast<size_t>(compute_domain[N - 1]),
        static_cast<size_t>(compute_domain[N - 2]),
        static_cast<size_t>(compute_domain[N - 3])};
#ifdef __AMP_CPU__
    if (CLAMP::is_cpu()) {
        launch_cpu_task(compute_domain, f);
        return;
    }
#endif
    const pfe_wrapper<N, Kernel> _pf(compute_domain, f);
    mcw_cxxamp_launch_kernel<pfe_wrapper<N, Kernel>, 3>(ext, NULL, _pf);
#else
#ifdef __AMP_CPU__
    int* foo1 = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//ND async_parallel_for_each, nontiled
template <int N, typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    extent<N> compute_domain, const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
    size_t compute_domain_size = 1;
    for(int i = 0 ; i < N ; i++)
    {
      if(compute_domain[i]<=0)
        throw invalid_compute_domain("Extent is less or equal than 0.");
      if (static_cast<size_t>(compute_domain[i]) > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
      compute_domain_size *= static_cast<size_t>(compute_domain[i]);
      if (compute_domain_size > 4294967295L)
        throw invalid_compute_domain("Extent size too large.");
    }

    size_t ext[3] = {static_cast<size_t>(compute_domain[N - 1]),
        static_cast<size_t>(compute_domain[N - 2]),
        static_cast<size_t>(compute_domain[N - 3])};
    const pfe_wrapper<N, Kernel> _pf(compute_domain, f);
    return completion_future(mcw_cxxamp_launch_kernel_async<pfe_wrapper<N, Kernel>, 3>(ext, NULL, _pf));
#else
#ifdef __AMP_CPU__
  int* foo1 = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
    auto bar = &pfe_wrapper<N, Kernel>::operator();
    auto qq = &index<N>::__cxxamp_opencl_index;
    int* foo = reinterpret_cast<int*>(&pfe_wrapper<N, Kernel>::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

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
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      launch_cpu_task(compute_domain, f);
      return;
  }
#endif
  size_t ext = compute_domain[0];
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D async_parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    extent<1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 1>(&ext, NULL, f));
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

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
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      launch_cpu_task(compute_domain, f);
      return;
  }
#endif
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
      static_cast<size_t>(compute_domain[0])};
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, NULL, f);
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D async_parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    extent<2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = {static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 2>(ext, NULL, f));
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

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
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      launch_cpu_task(compute_domain, f);
      return;
  }
#endif
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D async_parallel_for_each, nontiled
template <typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    extent<3> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[3] = {static_cast<size_t>(compute_domain[2]),
                   static_cast<size_t>(compute_domain[1]),
                   static_cast<size_t>(compute_domain[0])};
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 3>(ext, NULL, f));
#else //ifndef __GPU__
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

//1D parallel_for_each, tiled
template <int D0, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim0;
  static_assert( compute_domain.tile_dim0 <= 1024, "The maximum nuimber of threads in a tile is 1024");
  if(ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      Concurrency::Serialize s(nullptr);
      f.__cxxamp_serialize(s);
      CLAMP::enter_kernel();
      std::vector<std::thread> th(NTHREAD);
      for (int i = 0; i < NTHREAD; ++i)
          th[i] = std::thread(partitioned_task_tile<Kernel, D0>,
                              std::cref(f), std::cref(compute_domain), i);
      for (auto& t : th)
          if (t.joinable())
              t.join();
      f.__cxxamp_serialize(s);
      CLAMP::leave_kernel();
      return;
  }
#endif
  mcw_cxxamp_launch_kernel<Kernel, 1>(&ext, &tile, f);
#else //ifndef __GPU__
  tiled_index<D0> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//1D async_parallel_for_each, tiled
template <int D0, typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    tiled_extent<D0> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext = compute_domain[0];
  size_t tile = compute_domain.tile_dim0;
  static_assert( compute_domain.tile_dim0 <= 1024, "The maximum nuimber of threads in a tile is 1024");
  if(ext % tile != 0) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 1>(&ext, &tile, f));
#else //ifndef __GPU__
  tiled_index<D0> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

//2D parallel_for_each, tiled
template <int D0, int D1, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0, D1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { compute_domain.tile_dim1,
                     compute_domain.tile_dim0};
  static_assert( (compute_domain.tile_dim1 * compute_domain.tile_dim0)<= 1024, "The maximum nuimber of threads in a tile is 1024");
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      Concurrency::Serialize s(nullptr);
      f.__cxxamp_serialize(s);
      CLAMP::enter_kernel();
      std::vector<std::thread> th(NTHREAD);
      for (int i = 0; i < NTHREAD; ++i)
          th[i] = std::thread(partitioned_task_tile<Kernel, D0, D1>,
                              std::cref(f), std::cref(compute_domain), i);
      for (auto& t : th)
          if (t.joinable())
              t.join();
      f.__cxxamp_serialize(s);
      CLAMP::leave_kernel();
      return;
  }
#endif
  mcw_cxxamp_launch_kernel<Kernel, 2>(ext, tile, f);
#else //ifndef __GPU__
  tiled_index<D0, D1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//2D async_parallel_for_each, tiled
template <int D0, int D1, typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    tiled_extent<D0, D1> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  size_t ext[2] = { static_cast<size_t>(compute_domain[1]),
                    static_cast<size_t>(compute_domain[0])};
  size_t tile[2] = { compute_domain.tile_dim1,
                     compute_domain.tile_dim0};
  static_assert( (compute_domain.tile_dim1 * compute_domain.tile_dim0)<= 1024, "The maximum nuimber of threads in a tile is 1024");
  if((ext[0] % tile[0] != 0) || (ext[1] % tile[1] != 0)) {
    throw invalid_compute_domain("Extent can't be evenly divisble by tile size.");
  }
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 2>(ext, tile, f));
#else //ifndef __GPU__
  tiled_index<D0, D1> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

//3D parallel_for_each, tiled
template <int D0, int D1, int D2, typename Kernel>
__attribute__((noinline,used)) void parallel_for_each(
    tiled_extent<D0, D1, D2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
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
#ifdef __AMP_CPU__
  if (CLAMP::is_cpu()) {
      Concurrency::Serialize s(nullptr);
      f.__cxxamp_serialize(s);
      CLAMP::enter_kernel();
      std::vector<std::thread> th(NTHREAD);
      for (int i = 0; i < NTHREAD; ++i)
          th[i] = std::thread(partitioned_task_tile<Kernel, D0, D1, D2>,
                              std::cref(f), std::cref(compute_domain), i);
      for (auto& t : th)
          if (t.joinable())
              t.join();
      f.__cxxamp_serialize(s);
      CLAMP::leave_kernel();
      return;
  }
#endif
  mcw_cxxamp_launch_kernel<Kernel, 3>(ext, tile, f);
#else //ifndef __GPU__
  tiled_index<D0, D1, D2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
//3D async_parallel_for_each, tiled
template <int D0, int D1, int D2, typename Kernel>
__attribute__((noinline,used)) completion_future async_parallel_for_each(
    tiled_extent<D0, D1, D2> compute_domain,
    const Kernel& f) restrict(cpu,amp) {
#ifndef __GPU__
  if(compute_domain[0]<=0 || compute_domain[1]<=0 || compute_domain[2]<=0) {
    throw invalid_compute_domain("Extent is less or equal than 0.");
  }
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
  if (static_cast<size_t>(compute_domain[0]) * static_cast<size_t>(compute_domain[1]) * static_cast<size_t>(compute_domain[2]) > 4294967295L)
    throw invalid_compute_domain("Extent size too large.");
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
  return completion_future(mcw_cxxamp_launch_kernel_async<Kernel, 3>(ext, tile, f));
#else //ifndef __GPU__
  tiled_index<D0, D1, D2> this_is_used_to_instantiate_the_right_index;
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
#endif
}
#pragma clang diagnostic pop

} // namespace Concurrency
