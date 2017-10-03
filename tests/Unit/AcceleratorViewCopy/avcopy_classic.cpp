// RUN: %hc %s -o %t.out -lhc_am -L/opt/rocm/lib -lhsa-runtime64 && %t.out
//
// Test "classic" GPU pattern of H2D copies, followed by Kernels, followed by
// D2H.
// Test allows toggling explicit host-side syncs (via accelerator-view waits) vs
// relying on efficient GPU hardware dependencies.
#include <hc.hpp>
#include <hc_am.hpp>

#include "/opt/rocm/include/hsa/hsa.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

// Enable to show size and function parm info for each call to simpleTests
bool p_verbose = true;

// Set non-zero to run only one test.
int p_runtest = 0;

template<typename T, bool host_pinned>
std::unique_ptr<T[], decltype(hc::am_free)*> hostAlloc(
    const hc::accelerator& acc, std::size_t cnt)
{
    std::unique_ptr<T[], decltype(hc::am_free)*> p{
        host_pinned ? static_cast<T*>(
            hc::am_alloc(
                sizeof(T) * cnt,
                const_cast<hc::accelerator&>(acc),
                amHostPinned))
                    : new T[cnt],
        host_pinned ? hc::am_free
                    : [](void* p) { delete [] static_cast<T*>(p); return 0; }};

    return p;
}


// ****************************************************************************
void memcopy(
    const hc::accelerator_view& av,
    bool useAsyncCopy,
    const void* src,
    void* dst,
    std::size_t sizeBytes)
{
    if (useAsyncCopy) {
        const_cast<hc::accelerator_view&>(av).copy_async(src, dst, sizeBytes);
    }
    else {
        const_cast<hc::accelerator_view&>(av).copy(src, dst, sizeBytes);
    }
}

//---
// Test simple H2D copies, kernel, D2H copy.  Various knobs to control when
// synchronization occurs:
// acc : accelerator to run test on.  The default queue is used
//       (get_default_view).
// N = size of arrays to use, in elements.
// useAsyncCopy = use accelerator_view::copy_async for all copies. Else use
//                accelerator_view::copy.
// usePinnedHost = allocate pinned memory.  Else use malloc for allocations.
// syncAfter* - wait for accelerator_view to drain after H2D,Kernel,D2H.
//              Makes test easier since device-side dependency resolution
//              not used.
//
// Designed to stress a small number of simple smoke tests

int g_testnum = 0;

template<typename T, bool usePinnedHost>
bool simpleTest1(
    const hc::accelerator& acc,
    std::size_t N,
    bool useAsyncCopy,
    bool syncAfterH2D,
    bool syncAfterKernel,
    bool syncAfterD2H,
    int db = 0)
{
    ++g_testnum;
    if (p_runtest && (p_runtest != g_testnum)) return true;

    std::size_t Nbytes = N * sizeof(T);

    auto A_h = hostAlloc<T, usePinnedHost>(acc, N);
    auto B_h = hostAlloc<T, usePinnedHost>(acc, N);
    auto C_h = hostAlloc<T, usePinnedHost>(acc, N);

    std::unique_ptr<T[], decltype(hc::am_free)*> A_d{
        static_cast<T*>(
            hc::am_alloc(Nbytes, const_cast<hc::accelerator&>(acc), 0)),
        hc::am_free};
    std::unique_ptr<T[], decltype(hc::am_free)*> B_d{
        static_cast<T*>(
            hc::am_alloc(Nbytes, const_cast<hc::accelerator&>(acc), 0)),
        hc::am_free};
    std::unique_ptr<T[], decltype(hc::am_free)*> C_d{
        static_cast<T*>(
            hc::am_alloc(Nbytes, const_cast<hc::accelerator&>(acc), 0)),
        hc::am_free};

    if (!A_d || !B_d || !C_d) return false;

    // Initialize the host data:
    for (size_t i=0; i<N; i++) {
        A_h[i] = 3 + i;
        B_h[i] = 1 + i; // Phi
        C_h[i] = 1000 + i;
    }

    hc::accelerator_view av = acc.get_default_view();

    memcopy(av, useAsyncCopy, A_h.get(), A_d.get(), Nbytes);
    memcopy(av, useAsyncCopy, B_h.get(), B_d.get(), Nbytes);

    if (syncAfterH2D) av.wait();

    hc::parallel_for_each(
        av,
        hc::extent<1>(N),
        [A_d = A_d.get(),
            B_d = B_d.get(),
            C_d = C_d.get()](hc::index<1> idx) [[hc]] {
            C_d[idx[0]] = A_d[idx[0]] + B_d[idx[0]] ;
        });

    if (syncAfterKernel) av.wait();

    memcopy(av, useAsyncCopy, C_d.get(), C_h.get(), Nbytes);

    if (useAsyncCopy && !syncAfterD2H) return false;

    if (syncAfterD2H) av.wait();

    for (auto i = 0u; i != N; ++i) if (C_h[i] != A_h[i] + B_h[i]) return false;

    return true;
}


template<typename T>
bool test_size(const hc::accelerator& acc, std::size_t cnt)
{
    bool pass = true;

    const auto pinned_mem =
        static_cast<hsa_region_t*>(
            const_cast<hc::accelerator&>(acc).get_hsa_am_system_region());

    if (pinned_mem) {
        bool can_allocate = false;
        hsa_region_get_info(
            *pinned_mem, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &can_allocate);

        if (can_allocate) {
            std::size_t max_pinned_alloc_sz = 0u;
            hsa_region_get_info(
                *pinned_mem,
                HSA_REGION_INFO_ALLOC_MAX_SIZE,
                &max_pinned_alloc_sz);

            std::size_t max_pinned_cnt =
                std::min(max_pinned_alloc_sz / sizeof(T), cnt);
            //---
            // ASYNC code + pinned memory
            // Use async calls, but av.wait() after all the important steps:
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                true/*useAsyncCopy*/,
                true/*syncAfterH2D*/,
                true/*syncAfterKernel*/,
                true/*syncAfterD2H*/,
                0);

            // test H2D -> kernel dependency:
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                true/*useAsyncCopy*/,
                false/*syncAfterH2D*/,
                true/*syncAfterKernel*/,
                true/*syncAfterD2H*/,
                0);

            // Test H2D -> kernel -> D2H dependency.  If this fails, likely
            // indicates problem with D2H
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                true/*useAsyncCopy*/,
                false/*syncAfterH2D*/,
                false/*syncAfterKernel*/,
                true/*syncAfterD2H*/,
                0);

            // Note - don't test async with syncAfterD2H removed, we need this
            // sync before reading back on host.
            //---

            //---
            // Synchronous cases, pinned mem:
            // Sync copy, sync after all steps
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                false/*useAsyncCopy*/,
                true/*syncAfterH2D*/,
                true/*syncAfterKernel*/,
                true/*syncAfterD2H*/);

            // relax syncs between stages:
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                false/*useAsyncCopy*/,
                false/*syncAfterH2D*/,
                true/*syncAfterKernel*/,
                true/*syncAfterD2H*/);
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                false/*useAsyncCopy*/,
                false/*syncAfterH2D*/,
                false/*syncAfterKernel*/,
                true/*syncAfterD2H*/);
            pass = pass && simpleTest1<T, true>(
                acc,
                max_pinned_cnt,
                false/*useAsyncCopy*/,
                false/*syncAfterH2D*/,
                false/*syncAfterKernel*/,
                false/*syncAfterD2H*/);
            //---
        }
    }

    //---
    // Synchronous cases, unpinned mem:
    pass = pass && simpleTest1<T, false>(
        acc,
        cnt,
        false/*useAsyncCopy*/,
        true/*syncAfterH2D*/,
        true/*syncAfterKernel*/,
        true/*syncAfterD2H*/);

    // relax syncs between stages:
    pass = pass && simpleTest1<T, false>(
        acc,
        cnt,
        false/*useAsyncCopy*/,
        false/*syncAfterH2D*/,
        true/*syncAfterKernel*/,
        true/*syncAfterD2H*/);
    pass = pass && simpleTest1<T, false>(
        acc,
        cnt,
        false/*useAsyncCopy*/,
        false/*syncAfterH2D*/,
        false/*syncAfterKernel*/,
        true/*syncAfterD2H*/);
    pass = pass && simpleTest1<T, false>(
        acc,
        cnt,
        false/*useAsyncCopy*/,
        false/*syncAfterH2D*/,
        false/*syncAfterKernel*/,
        false/*syncAfterD2H*/);

    return pass;
}

int main()
{
    std::vector<hc::accelerator> accs = hc::accelerator::get_all();
    auto gpu_acc = std::find_if(
        accs.cbegin(),
        accs.cend(),
        [](const hc::accelerator& a) { return !a.get_is_emulated(); });
    if (gpu_acc == accs.cend()) return EXIT_FAILURE;

    bool pass = true;
    // medium:
    pass = pass && test_size<int>(*gpu_acc, 1024 * 256);

    // small:
    pass = pass && test_size<int>(*gpu_acc, 1024);

    // large:
    pass = pass && test_size<int>(*gpu_acc, 1024 * 1024 * 16);

    return EXIT_SUCCESS;
}