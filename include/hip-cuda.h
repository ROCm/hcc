#ifndef HIP_CUDA_H
#define HIP_CUDA_H

#include "cuda_runtime.h"
#include <assert.h>

typedef uint3 hc_uint3;

#define hcResetDefaultAccelerator() cudaDeviceReset()
#define hcMalloc(...) cudaMalloc(__VA_ARGS__)
#define hcMemcpy(...) cudaMemcpy(__VA_ARGS__)
#define hcFree(...) cudaFree(__VA_ARGS__)
#define hcMemset(...) cudaMemset(__VA_ARGS__)

#define __KERNEL __global__

#define HC_ASSERT(x) \
  assert(!x)

#define hcCreateLaunchParam2(blocks, threads) \
  blocks, threads

#define hcMemcpyHostToAccelerator cudaMemcpyHostToDevice
#define hcMemcpyAcceleratorToHost cudaMemcpyDeviceToHost

#define hcLaunchKernel(kernel, dim, ...) \
  grid_launch_parm lp; \
  kernel<<<dim>>>(lp, __VA_ARGS__)

#define DIM3(...) dim3(__VA_ARGS__)

#define GRID_LAUNCH_INIT(lp) \
    lp.gridDim.x = gridDim.x; \
    lp.gridDim.y = gridDim.y; \
    lp.gridDim.z = gridDim.z; \
    lp.groupDim.x = blockDim.x; \
    lp.groupDim.y = blockDim.y; \
    lp.groupDim.z = blockDim.z; \
    lp.groupId.x = blockIdx.x; \
    lp.groupId.y = blockIdx.y; \
    lp.groupId.z = blockIdx.z; \
    lp.threadId.x = threadIdx.x; \
    lp.threadId.y = threadIdx.y; \
    lp.threadId.z = threadIdx.z

#endif
