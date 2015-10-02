#ifndef hc_h
#define hc_h

#ifndef USE_CUDA
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct uint3
{
  int x,y,z;
} hc_uint3;

#else

#include "cuda_runtime.h"
#include <assert.h>

typedef uint3 hc_uint3;
#endif

typedef struct grid_launch_parm
{
  hc_uint3      gridDim;
  hc_uint3      groupDim;
  hc_uint3      groupId;
  hc_uint3      threadId;
  unsigned int  groupMemBytes;
  //accelerator_view av;
} grid_launch_parm;

#ifdef USE_CUDA
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

#else
#define __KERNEL __attribute__((hc_grid_launch))

typedef hc_uint3 dim3;

extern inline dim3 dim3_init(int x, int y, int z)
{
  dim3 tmp;
  tmp.x = x; tmp.y = y; tmp.z = z;
  return tmp;
}

extern inline dim3 dim3_eval(int x, ...)
{
  int y, z;
  va_list args;
  va_start(args, x);
  if(!(y = va_arg(args, int))) {
    va_end(args);
    return dim3_init(x, 1, 1);
  }
  if(!(z = va_arg(args, int))) {
    va_end(args);
    return dim3_init(x, y, 1);
  }
  va_end(args);
  return dim3_init(x, y, z);
}

#define DIM3(x, ...) \
  dim3_eval(x, __VA_ARGS__, NULL)

#define hcLaunchKernel(fn, lp, ...) \
  void __hcLaunchKernel_##fn(grid_launch_parm lp_arg, ...); \
  __hcLaunchKernel_##fn(lp, __VA_ARGS__)

#define HC_ASSERT(x) \
  assert(!x)

#define hcResetDefaultAccelerator()

extern inline int hcMalloc(void** ptr, size_t buf_size)
{
  *ptr = malloc(buf_size);
  return *ptr ? 0 : 1;
}

typedef enum
{
  hcMemcpyHostToAccelerator,
  hcMemcpyAcceleratorToHost
} hcMemcpyDir;

extern inline int hcMemcpy(void* dest, void* src, size_t buf_size, hcMemcpyDir memcpyDir)
{
  // TODO: Does direction matter?
  memcpy(dest, src, buf_size);
  if(dest)
    return 0;
  else return 1;
}

extern inline int hcFree(void* ptr)
{
  free(ptr);
  ptr = NULL;
  // TODO: How to handle errors?
  return 0;
}

extern inline int hcMemset(void* ptr, int value, size_t count)
{
  void * tmp = memset(ptr, value, count);
  if(tmp)
    return 0;
  else return 1;
}

extern inline grid_launch_parm hcCreateLaunchParam2(hc_uint3 gridDim, hc_uint3 groupDim)
{
  grid_launch_parm lp;

  lp.gridDim.x = gridDim.x;
  lp.gridDim.y = gridDim.y;

  lp.groupDim.x = groupDim.x;
  lp.groupDim.y = groupDim.y;

  return lp;
}

#define GRID_LAUNCH_INIT(lp)

#endif // USE_CUDA
#endif
