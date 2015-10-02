#ifndef HIP_CUDA_H
#define HIP_CUDA_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


#define hcResetDefaultAccelerator() cudaDeviceReset()
#define hcMalloc(...) cudaMalloc(__VA_ARGS__)
#define hcMemcpy(...) cudaMemcpy(__VA_ARGS__)
#define hcFree(...) cudaFree(__VA_ARGS__)
#define hcMemset(...) cudaMemset(__VA_ARGS__)
#define hcCreateChannelDesc() cudaCreateChannelDesc<float>() // TODO: Implement templates
#define hcMallocPitch(...) cudaMallocPitch(__VA_ARGS__)
#define hcMallocArray(...) cudaMallocArray(__VA_ARGS__)
#define hcMemcpy2D(...) cudaMemcpy2D(__VA_ARGS__)
#define hcMemcpy2DToArray(...) cudaMemcpy2DToArray(__VA_ARGS__)
#define hcBindTextureToArray(...) cudaBindTextureToArray(__VA_ARGS__)
#define hcUnbindTexture(...) cudaUnbindTexture(__VA_ARGS__)
#define hcArray cudaArray
#define hcChannelFormatDesc cudaChannelFormatDesc
#define texture texture<float,2> // TODO: Implement templates
#define hcFilterModePoint cudaFilterModePoint

#define CUDA_SAFE_CALL(x) checkCudaErrors(x)
#define CUT_CHECK_ERROR(x) getLastCudaError(x)

#define __KERNEL __global__

#define HC_ASSERT(x) \
  assert(!x)

#define hcCreateLaunchParam2(blocks, threads) \
  blocks, threads

#define hcMemcpyHostToAccelerator cudaMemcpyHostToDevice
#define hcMemcpyAcceleratorToHost cudaMemcpyDeviceToHost
#define hcMemcpyAcceleratorToAccelerator cudaMemcpyDeviceToDevice

#define hcLaunchKernel(kernel, grid, block, ...) \
  grid_launch_parm lp; \
  lp.gridDim.x = grid.x; \
  lp.gridDim.y = grid.y; \
  lp.gridDim.z = grid.z; \
  lp.groupDim.x = block.x; \
  lp.groupDim.y = block.y; \
  lp.groupDim.z = block.z; \
  kernel<<<grid, block>>>(lp, __VA_ARGS__)

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

#define SQRTF(x) sqrtf(x)

#endif
