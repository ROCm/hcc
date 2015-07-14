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
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#include <thread>
#endif

#include <amp.h>
#include <kalmar_runtime.h>

namespace Kalmar {

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#define SSIZE 1024 * 10
static const unsigned int NTHREAD = std::thread::hardware_concurrency();
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

template <typename Kernel>
class CPUKernelRAII
{
    const std::shared_ptr<KalmarQueue> pQueue;
    const Kernel& f;
    std::vector<std::thread> th;
public:
    CPUKernelRAII(const std::shared_ptr<KalmarQueue> pQueue, const Kernel& f)
        : pQueue(pQueue), f(f), th(NTHREAD) {
        Kalmar::CPUVisitor vis(pQueue);
        Kalmar::Serialize s(&vis);
        f.__cxxamp_serialize(s);
        CLAMP::enter_kernel();
    }
    std::thread& operator[](int i) { return th[i]; }
    ~CPUKernelRAII() {
        for (auto& t : th)
            if (t.joinable())
                t.join();
        Kalmar::CPUVisitor vis(pQueue);
        Kalmar::Serialize ss(&vis);
        f.__cxxamp_serialize(ss);
        CLAMP::leave_kernel();
    }
};

template <typename Kernel, int N>
void launch_cpu_task(const accelerator_view& av, Kernel const& f,
                     extent<N> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(av.pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(partitioned_task<Kernel, N>, std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     tiled_extent<D0> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(partitioned_task_tile<Kernel, D0>,
                             std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0, int D1>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     tiled_extent<D0, D1> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(partitioned_task_tile<Kernel, D0, D1>,
                             std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0, int D1, int D2>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     tiled_extent<D0, D1, D2> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(partitioned_task_tile<Kernel, D0, D1, D2>,
                             std::cref(f), std::cref(compute_domain), i);
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
    return out;
}

template <typename Kernel>
static void append_kernel(std::shared_ptr<KalmarQueue> av, const Kernel& f, void* kernel)
{
  Kalmar::BufferArgumentsAppender vis(av, kernel);
  Kalmar::Serialize s(&vis);
  f.__cxxamp_serialize(s);
}

static std::set<std::string> __mcw_cxxamp_kernels;
template<typename Kernel, int dim_ext>
inline std::shared_future<void>*
mcw_cxxamp_launch_kernel_async(const accelerator_view& av, size_t *ext,
  size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string transformed_kernel_name =
          mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(transformed_kernel_name, av.pQueue.get());
  }
  append_kernel(av.pQueue, f, kernel);
  return static_cast<std::shared_future<void>*>(av.pQueue->LaunchKernelAsync(kernel, dim_ext, ext, local_size));
#endif
}

template<typename Kernel>
inline void* mcw_cxxamp_get_kernel(const accelerator_view& av, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  std::string transformed_kernel_name =
      mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
  kernel = CLAMP::CreateKernel(transformed_kernel_name, av.pQueue.get());
  return kernel;
#else
  return NULL;
#endif
}

template<typename Kernel, int dim_ext>
inline
void mcw_cxxamp_execute_kernel_with_dynamic_group_memory(
  const accelerator_view& av, size_t *ext, size_t *local_size,
  const Kernel& f, void *kernel, size_t dynamic_group_memory_size) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  append_kernel(av.pQueue, f, kernel);
  av.pQueue->LaunchKernelWithDynamicGroupMemory(kernel, dim_ext, ext, local_size, dynamic_group_memory_size);
#endif // __KALMAR_ACCELERATOR__
}

template<typename Kernel, int dim_ext>
inline
void mcw_cxxamp_launch_kernel(const accelerator_view& av, size_t *ext,
                              size_t *local_size, const Kernel& f) restrict(cpu,amp) {
#if __KALMAR_ACCELERATOR__ != 1
  if (av.get_accelerator().get_device_path() == L"cpu") {
    throw runtime_exception(__errorMsg_UnsupportedAccelerator, E_FAIL);
  }
  //Invoke Kernel::__cxxamp_trampoline as an kernel
  //to ensure functor has right operator() defined
  //this triggers the trampoline code being emitted
  // FIXME: implicitly casting to avoid pointer to int error
  int* foo = reinterpret_cast<int*>(&Kernel::__cxxamp_trampoline);
  void *kernel = NULL;
  {
      std::string transformed_kernel_name =
          mcw_cxxamp_fixnames(f.__cxxamp_trampoline_name());
      kernel = CLAMP::CreateKernel(transformed_kernel_name, av.pQueue.get());
  }
  append_kernel(av.pQueue, f, kernel);
  av.pQueue->LaunchKernel(kernel, dim_ext, ext, local_size);
#endif // __KALMAR_ACCELERATOR__
}

} // namespace Kalmar

