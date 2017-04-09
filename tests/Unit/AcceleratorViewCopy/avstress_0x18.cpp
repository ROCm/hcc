// RUN: %hc %s -o %t.out -lhc_am -L/opt/rocm/lib -lhsa-runtime64 -DRUNMASK=0x18 && HCC_SERIALIZE_KERNEL=0x3 HCC_SERIALIZE_COPY=0x3 %t.out
#include <hc.hpp>
#include <hc_am.hpp>

#include "/opt/rocm/include/hsa/hsa.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>


#if !defined(RUNMASK)
    #define RUNMASK 0x18
#endif

#if !defined(ITERS)
    #define ITERS 2000
#endif

#define HostToDeviceCopyTest 0x1
#define DeviceToDeviceCopyTest 0x2
#define DeviceToHostCopyTest 0x4
#define HostToDeviceAsyncCopyTest 0x8
#define DeviceToHostAsyncCopyTest 0x10

int main()
{
    constexpr std::size_t N = 1024 * 1024;

    std::vector <hc::accelerator> accs = hc::accelerator::get_all();
    auto acc = std::find_if(
        accs.cbegin(),
        accs.cend(),
        [](const hc::accelerator& a) { return !a.get_is_emulated(); });
    if (acc == accs.cend()) return EXIT_FAILURE;

    const auto pinned_mem = static_cast<hsa_region_t*>(
        const_cast<hc::accelerator&>(*acc).get_hsa_am_system_region());

    bool can_allocate = false;
    hsa_region_get_info(
        *pinned_mem, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &can_allocate);

    if (!can_allocate) return EXIT_FAILURE;

    std::size_t max_pinned_alloc_sz = 0u;
    hsa_region_get_info(
        *pinned_mem, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_pinned_alloc_sz);

    std::size_t max_pinned_cnt = std::min(max_pinned_alloc_sz / sizeof(int), N);
    std::size_t max_pinned_sz = max_pinned_cnt * sizeof(int);

    std::vector<int> A(max_pinned_cnt, 3);
    std::vector<int> B(max_pinned_cnt, 1);
    std::vector<int> C;

    std::unique_ptr<int[], decltype(hc::am_free) * > A_h{
        hc::am_alloc(
            max_pinned_sz, const_cast<hc::accelerator&>(*acc), amHostPinned),
        hc::am_free};
    std::copy_n(A.cbegin(), A.size(), A_h.get());

    std::unique_ptr<int[], decltype(hc::am_free) * > B_h{
        hc::am_alloc(
            max_pinned_sz, const_cast<hc::accelerator&>(*acc), amHostPinned),
        hc::am_free};
    std::copy_n(B.begin(), B.size(), B_h.get());

    std::unique_ptr<int[], decltype(hc::am_free) * > C_h{
        hc::am_alloc(
            max_pinned_sz, const_cast<hc::accelerator&>(*acc), amHostPinned),
        hc::am_free};

    std::unique_ptr<int[], decltype(hc::am_free) * > A_d{
        hc::am_alloc(max_pinned_sz, const_cast<hc::accelerator&>(*acc), 0x0),
        hc::am_free};
    std::unique_ptr<int[], decltype(hc::am_free) * > B_d{
        hc::am_alloc(max_pinned_sz, const_cast<hc::accelerator&>(*acc), 0x0),
        hc::am_free};
    std::unique_ptr<int[], decltype(hc::am_free) * > C_d{
        hc::am_alloc(max_pinned_sz, const_cast<hc::accelerator&>(*acc), 0x0),
        hc::am_free};

    // RUNMASK should be #defined as input to compilation:
    constexpr auto tests_to_run = RUNMASK;
    constexpr auto iter_cnt = ITERS;

    if (tests_to_run & HostToDeviceCopyTest) {
        for (auto i = 0u; i != iter_cnt; ++i) {
            acc->get_default_view().copy(A.data(), A_d.get(), max_pinned_sz);
        }
        acc->get_default_view().copy(A_d.get(), A_h.get(), max_pinned_sz);
        if (!std::equal(A.cbegin(), A.cend(), A_h.get())) {
            return EXIT_FAILURE;
        }
    }

    if (tests_to_run & DeviceToDeviceCopyTest) {
        acc->get_default_view().copy(B.data(), B_d.get(), max_pinned_sz);
        for (auto i = 0u; i != iter_cnt; ++i) {
            hc::parallel_for_each(
                hc::extent<1>(max_pinned_cnt),
                [A_d = A_d.get(),
                 B_d = B_d.get(),
                 C_d = C_d.get()](hc::index<1> idx) [[hc]] {
                    C_d[idx[0]] = A_d[idx[0]] + B_d[idx[0]];
                });
        }
        acc->get_default_view().copy(C_d.get(), C_h.get(), max_pinned_sz);
        for (auto i = 0u; i != max_pinned_cnt; ++i) {
            if (C_h[i] != A[i] + B[i]) return EXIT_FAILURE;
        }
    }

    if (tests_to_run & DeviceToHostCopyTest) {
        acc->get_default_view().copy(B.data(), B_d.get(), max_pinned_sz);
        for (auto i = 0u; i != iter_cnt; ++i) {
            acc->get_default_view().copy(B_d.get(), B_h.get(), max_pinned_sz);
        }
        if (!std::equal(B.cbegin(), B.cend(), B_h.get())) {
            return EXIT_FAILURE;
        }
    }

    std::vector <hc::completion_future> cfs;

    if (tests_to_run & HostToDeviceAsyncCopyTest) {
        for (decltype(cfs.size()) i = 0; i != iter_cnt; ++i) {
            cfs.push_back(acc->get_default_view().copy_async(
                A_h.get(), A_d.get(), max_pinned_sz));
        }
        for (auto&& cf : cfs) cf.wait();

        acc->get_default_view().copy(A_d.get(), A_h.get(), max_pinned_sz);
        if (!std::equal(A.cbegin(), A.cend(), A_h.get())) {
            return EXIT_FAILURE;
        }
    }

    if (tests_to_run & DeviceToHostAsyncCopyTest) {
        acc->get_default_view().copy(B.data(), B_d.get(), max_pinned_sz);
        for (decltype(cfs.size()) i = 0; i != iter_cnt; ++i) {
            cfs.push_back(acc->get_default_view().copy_async(
                B_d.get(), B_h.get(), max_pinned_sz));
        }
        for (auto&& cf : cfs) cf.wait();

        if (!std::equal(B.cbegin(), B.cend(), B_h.get())) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}