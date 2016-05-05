//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hc_defines.h"
#include "kalmar_runtime.h"
#include "kalmar_serialize.h"

namespace Kalmar {
template <int D0, int D1=0, int D2=0> class tiled_extent;

#if __KALMAR_ACCELERATOR__ == 2 || __KALMAR_CPU__ == 2
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
        CPUVisitor vis(pQueue);
        Serialize s(&vis);
        f.__cxxamp_serialize(s);
        CLAMP::enter_kernel();
    }
    std::thread& operator[](int i) { return th[i]; }
    ~CPUKernelRAII() {
        for (auto& t : th)
            if (t.joinable())
                t.join();
        CPUVisitor vis(pQueue);
        Serialize ss(&vis);
        f.__cxxamp_serialize(ss);
        CLAMP::leave_kernel();
    }
};

#endif

}
