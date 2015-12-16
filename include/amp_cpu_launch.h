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
using namespace Kalmar::CLAMP;
static const unsigned int NTHREAD = std::thread::hardware_concurrency();

template <typename Kernel>
class CPUKernelRAII
{
    const std::shared_ptr<Kalmar::KalmarQueue> pQueue;
    const Kernel& f;
    std::vector<std::thread> th;
public:
    CPUKernelRAII(const std::shared_ptr<Kalmar::KalmarQueue> pQueue, const Kernel& f)
        : pQueue(pQueue), f(f), th(NTHREAD) {
        Kalmar::CPUVisitor vis(pQueue);
        Kalmar::Serialize s(&vis);
        f.__cxxamp_serialize(s);
        enter_kernel();
    }
    std::thread& operator[](int i) { return th[i]; }
    ~CPUKernelRAII() {
        for (auto& t : th)
            if (t.joinable())
                t.join();
        Kalmar::CPUVisitor vis(pQueue);
        Kalmar::Serialize ss(&vis);
        f.__cxxamp_serialize(ss);
        leave_kernel();
    }
};

#endif

}
