
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <iostream>
#include <iomanip>
#include <vector>

// An end-to-end example of dynamic group segment usage

template<typename T>
bool test(size_t N, size_t groupElements) {
    bool ret = true;

    using namespace hc;

    std::vector<T> A_h(N), B_h(N);

    // initialze input
    for (int i = 0; i < N; ++i) {
        A_h[i] = 3.146f + i; // Pi
        B_h[i] = 1.618f + i; // Phi
    }

    // copy data from host to device
    array<T, 1> A_d(N, std::begin(A_h)), B_d(N, std::begin(B_h)), C_d(N);

    // calculate the amount of dynamic shared memory required
    size_t groupMemBytes = groupElements * sizeof(T);

    // launch kernel with dynamic shared memory
    extent<1> ex(N);
    tiled_extent<1> te = ex.tile_with_dynamic(256, groupMemBytes);
    parallel_for_each(te, [&,groupElements](tiled_index<1>& tidx) [[hc]] {
        size_t gid = tidx.global[0];
        size_t tid = tidx.local[0];

        T* sdata = (T*) get_dynamic_group_segment_base_pointer();

        // initialize dynamic shared memory
        if (tid < groupElements) {
            sdata[tid] = static_cast<T>(tid);
        }

        // prefix sum inside dynamic shared memory
        if (groupElements >= 512) {
            if (tid >= 256) { sdata[tid] += sdata[tid - 256]; } tidx.barrier.wait();
        }
        if (groupElements >= 256) {
            if (tid >= 128) { sdata[tid] += sdata[tid - 128]; } tidx.barrier.wait();
        }
        if (groupElements >= 128) {
            if (tid >= 64) { sdata[tid] += sdata[tid - 64]; } tidx.barrier.wait();
        }
        if (groupElements >= 64) { sdata[tid] += sdata[tid - 32]; } tidx.barrier.wait();
        if (groupElements >= 32) { sdata[tid] += sdata[tid - 16]; } tidx.barrier.wait();
        if (groupElements >= 16) { sdata[tid] += sdata[tid - 8]; } tidx.barrier.wait();
        if (groupElements >= 8) { sdata[tid] += sdata[tid - 4]; } tidx.barrier.wait();
        if (groupElements >= 4) { sdata[tid] += sdata[tid - 2]; } tidx.barrier.wait();
        if (groupElements >= 2) { sdata[tid] += sdata[tid - 1]; } tidx.barrier.wait();
    
        C_d[gid] = A_d[gid] + B_d[gid] + sdata[tid % groupElements];

    }).wait();

    // copy data from device to host
    std::vector<T> C_h = C_d;

    // verify
    for (size_t i = 0; i < N; ++i) {
        size_t tid = (i % groupElements);
        T sumFromSharedMemory = static_cast<T>(tid * (tid + 1) / 2);
        T expected = A_h[i] + B_h[i] + sumFromSharedMemory;
        if (C_h[i] != expected) {
           std::cout << std::fixed << std::setprecision(32);
           std::cout << "At " << i << std::endl;
           std::cout << "  Computed:" << C_h[i] << std::endl;
           std::cout << "  Expected:" << expected << std::endl;
           std::cout << sumFromSharedMemory << std::endl;
           std::cout << A_h[i] << std::endl;
           std::cout << B_h[i] << std::endl;

           std::cout << "Failed at index: " << i << std::endl;

           ret = false;
           break;
        }
    }

    return ret;
}

int main() {
    bool ret = true;

  // The test case is only workable on LC backend as of now
  // because on HSAIL backend there is no way to check the size of
  // group segment.

  // Skip the test in case we are not using LC backend
#if __hcc_backend__ == HCC_BACKEND_AMDGPU
    ret &= test<float>(1024, 4);
    ret &= test<float>(1024, 8);
    ret &= test<float>(1024, 16);
    ret &= test<float>(1024, 32);
    ret &= test<float>(1024, 64);

    ret &= test<float>(65536, 4);
    ret &= test<float>(65536, 8);
    ret &= test<float>(65536, 16);
    ret &= test<float>(65536, 32);
    ret &= test<float>(65536, 64);

    ret &= test<double>(1024, 4);
    ret &= test<double>(1024, 8);
    ret &= test<double>(1024, 16);
    ret &= test<double>(1024, 32);
    ret &= test<double>(1024, 64);

    ret &= test<double>(65536, 4);
    ret &= test<double>(65536, 8);
    ret &= test<double>(65536, 16);
    ret &= test<double>(65536, 32);
    ret &= test<double>(65536, 64);
#endif

    return !(ret == true);
}

