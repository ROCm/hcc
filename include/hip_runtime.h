#pragma once

#include <hip.h>


#ifndef USE_CUDA
#include <iostream>
#include <vector>

#define __global__
#define __host__

#define hipThreadIdx_x (lp.threadId.x)
#define hipThreadIdx_y (lp.threadId.y)
#define hipThreadIdx_z (lp.threadId.z)

#define hipBlockIdx_x  (lp.groupId.x)
#define hipBlockIdx_y  (lp.groupId.y)
#define hipBlockIdx_z  (lp.groupId.z)

#define hipBlockDim_x  (lp.groupDim.x)
#define hipBlockDim_y  (lp.groupDim.y)
#define hipBlockDim_z  (lp.groupDim.z)

#define hipGridDim_x  (lp.gridDim.x)
#define hipGridDim_y  (lp.gridDim.y)
#define hipGridDim_z  (lp.gridDim.z)


struct ihipStream_t;
typedef struct ihipStream_t *hipStream_t;


typedef enum hipError_t {
   hipSuccess = 0
} hipError_t;


extern "C" {

grid_launch_parm hipCreateLaunchParam2(hc_uint3 gridDim, hc_uint3 groupDim);

grid_launch_parm hipCreateLaunchParam3(hc_uint3 gridDim, hc_uint3 groupDim,
                                       int groupMemBytes);

grid_launch_parm hipCreateLaunchParam4(hc_uint3 gridDim, hc_uint3 groupDim,
                                       int groupMemBytes, hipStream_t stream);

hipError_t hipStreamCreate(hipStream_t *stream);

hipError_t hipStreamSynchronize(hipStream_t stream=nullptr);

hipError_t hipStreamDestroy(hipStream_t stream);

hipError_t hipSetDevice(int device);

hipError_t hipGetDevice(int *device);

hipError_t hipGetDeviceCount(int *count);

int hipDeviceSynchronize(void);

hipError_t hipMemcpyAsync(void *dst, const void *src,
                          size_t  count,
                          hipMemcpyKind kind,
                          hipStream_t stream=nullptr);

hipError_t hipMemsetAsync(void *dst, int value, size_t count,
                          hipStream_t stream=nullptr);
} // extern "C"
#endif // #ifndef USE_CUDA
