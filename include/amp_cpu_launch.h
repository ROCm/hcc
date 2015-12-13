//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <kalmar_defines.h>
#include <kalmar_runtime.h>
#include <kalmar_serialize.h>

namespace Concurrency {
template <int D0, int D1=0, int D2=0> class tiled_extent;

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
class accelerator_view;
struct accelerator_view_helper
{
  static inline const accelerator_view create_view(std::shared_ptr<Kalmar::KalmarQueue> pQueue);
  static inline std::shared_ptr<Kalmar::KalmarQueue> getPQueue(const std::shared_ptr<Kalmar::KalmarQueue> & av);
};

template <typename Kernel, int N>
  void partitioned_task(const Kernel& ker, const extent<N>& ext, int part);

template <typename Kernel, int D0>
  void partitioned_task_tile(Kernel const& f, tiled_extent<D0> const& ext, int part);

template <typename Kernel, int D0, int D1>
  void partitioned_task_tile(Kernel const& f, tiled_extent<D0, D1> const& ext, int part);

template <typename Kernel, int D0, int D1, int D2>
  void partitioned_task_tile(Kernel const& f, tiled_extent<D0, D1, D2> const& ext, int part);

#endif
}

/** \cond HIDDEN_SYMBOLS */
namespace Kalmar {
#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
#define SSIZE 1024 * 10
static const unsigned int NTHREAD = std::thread::hardware_concurrency();

template <int N>
  using extent = Concurrency::extent<N>;

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

// FIXME: need to resolve the dependency to extent
template <typename Kernel, int N>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& av, Kernel const& f,
                     extent<N> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(Concurrency::accelerator_view_helper::getPQueue(av), f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(Concurrency::partitioned_task<Kernel, N>, std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     Concurrency::tiled_extent<D0> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(Concurrency::partitioned_task_tile<Kernel, D0>,
                             std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0, int D1>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     Concurrency::tiled_extent<D0, D1> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(Concurrency::partitioned_task_tile<Kernel, D0, D1>,
                             std::cref(f), std::cref(compute_domain), i);
}

template <typename Kernel, int D0, int D1, int D2>
void launch_cpu_task(const std::shared_ptr<KalmarQueue>& pQueue, Kernel const& f,
                     Concurrency::tiled_extent<D0, D1, D2> const& compute_domain)
{
    CPUKernelRAII<Kernel> obj(pQueue, f);
    for (int i = 0; i < NTHREAD; ++i)
        obj[i] = std::thread(Concurrency::partitioned_task_tile<Kernel, D0, D1, D2>,
                             std::cref(f), std::cref(compute_domain), i);
}
#endif

}
