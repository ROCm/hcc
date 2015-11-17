// XXX(Yan-Ming): borrow from AMD's hip repo
#pragma once

#ifndef USE_CUDA
#include <iostream>
#include <vector>

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <grid_launch.h>

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
   hipSuccess = 0,
   hipErrorUnknown
} hipError_t;

typedef enum {
  hipChannelFormatKindSigned = 0,
  hipChannelFormatKindUnsigned,
  hipChannelFormatKindFloat,
  hipChannelFormatKindNone

} hipChannelFormatKind;

typedef struct hipChannelFormatDesc_s {
  int x;
  int y;
  int z;
  int w;
  hipChannelFormatKind f;
} hipChannelFormatDesc;

typedef struct hipArray_s {
  unsigned int width;
  unsigned int height;
  float* data; //FIXME: generalize this
} hipArray;

typedef enum {
  hipMemcpyHostToDevice,
  hipMemcpyDeviceToHost,
  hipMemcpyDeviceToDevice
} hipMemcpyKind;

typedef enum {
  hipAddressModeWrap,
  hipAddressModeClamp,
  hipAddressModeMirror,
  hipAddressModeBorder
} hipTextureAddressMode;

typedef enum {
  hipFilterModePoint,
  hipFilterModeLinear
} hipTextureFilterMode;

// TODO: Templatize
//template <typename T, int const dim>
typedef struct textureReference {
  hipTextureAddressMode addressMode[3];
  hipChannelFormatDesc channelDesc;
  hipTextureFilterMode filterMode;
  int normalized;
  int sRGB;
  hipArray* data;

  textureReference() {}
  textureReference(const textureReference &tex) {
    addressMode[0] = tex.addressMode[0];
    addressMode[1] = tex.addressMode[1];
    addressMode[2] = tex.addressMode[2];
    channelDesc = tex.channelDesc;
    filterMode = tex.filterMode;
    normalized = tex.normalized;
    sRGB = tex.sRGB;
    data = tex.data;
  }

} texture;

extern "C" {

grid_launch_parm hipCreateLaunchParam(uint3 gridDim, uint3 groupDim,
                                      int groupMemBytes, hipStream_t stream);

#define hipLaunchKernel(fn, grid, block, groupMemBytes, av, ...) \
  {grid_launch_parm lp = hipCreateLaunchParam(grid, block, groupMemBytes, av); \
  hc::completion_future cf; \
  lp.cf = &cf; \
  fn(lp, __VA_ARGS__); \
  lp.cf->wait(); }

hipError_t hipMalloc(void** ptr, size_t size);

hipError_t hipMallocHost(void** ptr, size_t buf_size);

// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);

// TODO: Improve this to include other things from desc
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height = 0, unsigned int flags = 0);

hipError_t hipMemcpy(void* dest, const void* src, size_t size, hipMemcpyKind kind);

// dpitch, spitch, and width in bytes
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);


// wOffset, width, and spitch in bytes
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                              size_t spitch, size_t width, size_t height, hipMemcpyKind kind);


//template <typename T, const int dim>
hipError_t hipBindTextureToArray(texture& tex, hipArray* array);

//template <typename T, const int dim>
hipError_t hipUnbindTexture(texture & tex);

//template <typename T>
#define tex2D(tex, dx, dy) \
  tex.data->data[(unsigned int)dx + (unsigned int)dy*(tex.data->width)]

hipError_t hipFree(void* ptr);

hipError_t hipFreeHost(void* ptr);

hipError_t hipMemset(void* ptr, int value, size_t count);

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

// Math MACROS
#define SQRTF(x) hc::precise_math::sqrtf(x)

} // extern "C"

// C++
hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);
hipChannelFormatDesc hipCreateChannelDesc();

typedef uint3 dim3;

dim3 DIM3(int x);
dim3 DIM3(int x, int y);
dim3 DIM3(int x, int y, int z);

// Compatibility Macros
#define GRID_LAUNCH_INIT(lp)
#define CUT_CHECK_ERROR(x)

// Debug
#define HIP_ASSERT(x) \
  assert(!x)

#define CUDA_SAFE_CALL(fn) if(fn) fprintf(stderr, "Error in %s:%d\n", __FILE__, __LINE__)

#endif // #ifndef USE_CUDA
