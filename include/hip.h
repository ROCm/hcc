#ifndef hc_h
#define hc_h

#ifdef USE_CUDA
#include "cuda_runtime.h"
#include <assert.h>

typedef uint3 hc_uint3;

#else

typedef struct uint3
{
  int x,y,z;
} hc_uint3;
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
#include "hip-cuda.h"

#else
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <hc.hpp>

#define __KERNEL __attribute__((hc_grid_launch))

// Prevent host-side compilation from compiler errors
#ifndef __GPU__
#define hc_barrier(n)
#endif

#define __GROUP static __attribute__((address_space(3)))
#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

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

#define CUDA_SAFE_CALL(fn) if(fn) fprintf(stderr, "Error in %s:%d\n", __FILE__, __LINE__)

extern inline int hcDeviceSynchronize()
{
  hc::accelerator().get_default_view().wait();
  return 0;
}

extern inline int hcMalloc(void** ptr, size_t buf_size)
{
  *ptr = malloc(buf_size);
  return *ptr ? 0 : 1;
}

typedef enum
{
  hcChannelFormatKindSigned = 0,
  hcChannelFormatKindUnsigned,
  hcChannelFormatKindFloat,
  hcChannelFormatKindNone

} hcChannelFormatKind;

typedef struct hcChannelFormatDesc_s
{
  int x;
  int y;
  int z;
  int w;
  hcChannelFormatKind f;
} hcChannelFormatDesc;

extern inline hcChannelFormatDesc hcCreateChannelDesc(int x, int y, int z, int w, hcChannelFormatKind f)
{
  hcChannelFormatDesc cd;
  cd.x = x; cd.y = y; cd.z = z; cd.w = w;
  cd.f = f;
  return cd;
}

extern inline hcChannelFormatDesc hcCreateChannelDesc()
{
  return hcCreateChannelDesc(0, 0, 0, 0, hcChannelFormatKindFloat);
}

// width in bytes
extern inline int hcMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height)
{
  if(width == 0 || height == 0) return 1;
  *pitch = ((((int)width-1)/128) + 1)*128;
  *ptr = malloc((*pitch)*height);
  return *ptr ? 0 : 1;
}

typedef struct hcArray_s
{
  unsigned int width;
  unsigned int height;
  float* data; //FIXME: generalize this
} hcArray;

// TODO: Improve this to include other things from desc
extern inline int hcMallocArray(hcArray** array, const hcChannelFormatDesc* desc,
                                  size_t width, size_t height = 0, unsigned int flags = 0)
{
  *array = (hcArray*)malloc(sizeof(hcArray));
  if(desc->f == hcChannelFormatKindFloat)
  {
    array[0]->data = (float*)malloc(width*height*sizeof(float));
  }
  array[0]->width = width;
  array[0]->height = height;
  return *array ? 0 : 1;
}

typedef enum
{
  hcMemcpyHostToAccelerator,
  hcMemcpyAcceleratorToHost,
  hcMemcpyAcceleratorToAccelerator
} hcMemcpyKind;

// dpitch, spitch, and width in bytes
extern inline int hcMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hcMemcpyKind kind)
{
  if(width > dpitch || width > spitch)
    return 1;
// FIXME: generalize float
  int dp_sz = dpitch/sizeof(float);
  int sp_sz = spitch/sizeof(float);
  for(int i = 0; i < height; ++i)
  {
    memcpy((float*)dst + i*dp_sz, (float*)src + i*sp_sz, width);
  }
  return dst ? 0 : 1;
}

// wOffset, width, and spitch in bytes
extern inline int hcMemcpy2DToArray(hcArray* dst, size_t wOffset, size_t hOffset, const void* src,
                                    size_t spitch, size_t width, size_t height, hcMemcpyKind kind)
{
  if((wOffset + width > (dst->width * sizeof(float))) || width > spitch)
  {
    fprintf(stderr, "wOffset: %lu, width: %lu, dst->width: %u, spitch: %lu\n", wOffset, width, dst->width, spitch);
    return 1;
  }

// FIXME: generalize type
  int src_w = width/sizeof(float);
  int dst_w = dst->width;

  for(int i = 0; i < height; ++i)
  {
    memcpy((float*)dst->data + i*dst_w, (float*)src + i*src_w, width);
  }
  return 0;
}

typedef enum
{
  hcAddressModeWrap,
  hcAddressModeClamp,
  hcAddressModeMirror,
  hcAddressModeBorder
} hcTextureAddressMode;

typedef enum
{
  hcFilterModePoint,
  hcFilterModeLinear
} hcTextureFilterMode;

// TODO: Templatize
//template <typename T, int const dim>
typedef struct textureReference
{
  hcTextureAddressMode addressMode[3];
  hcChannelFormatDesc channelDesc;
  hcTextureFilterMode filterMode;
  int normalized;
  int sRGB;
  hcArray* data;

  textureReference() {}
  textureReference(const textureReference &tex)
  {
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

//template <typename T, const int dim>
extern inline int hcBindTextureToArray(texture& tex, hcArray* array)
{
  tex.data = array;
  return 0;
}

//template <typename T, const int dim>
extern inline int hcUnbindTexture(texture & tex)
{
  tex.data = NULL;
  return 0;
}

//template <typename T>
#define tex2D(tex, dx, dy) \
  tex.data->data[(unsigned int)dx + (unsigned int)dy*(tex.data->width)]

extern inline int hcMemcpy(void* dest, void* src, size_t buf_size, hcMemcpyKind kind)
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
  lp.gridDim.z = gridDim.z;

  lp.groupDim.x = groupDim.x;
  lp.groupDim.y = groupDim.y;
  lp.groupDim.z = groupDim.z;

  return lp;
}

#define GRID_LAUNCH_INIT(lp)

#define CUT_CHECK_ERROR(x)

#endif // USE_CUDA

#endif
