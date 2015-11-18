#include <hip_runtime.h>
#ifndef USE_CUDA

namespace hip_runtime {
  static std::vector<hc::accelerator> *g_device = nullptr;
  static int _the_device = 1;
}


struct ihipStream_t {
  hc::accelerator_view av;
  ihipStream_t(hc::accelerator_view av) : av(av) { };
};


void __attribute__ ((constructor)) hip_init() {
  hip_runtime::g_device = new std::vector<hc::accelerator>(hc::accelerator().get_all());
  // use HSA device by default
  hip_runtime::_the_device = 1;
}

grid_launch_parm hipCreateLaunchParam(uint3 gridDim, uint3 groupDim,
                                      int groupMemBytes, hipStream_t stream) {
  grid_launch_parm lp;

  lp.gridDim.x = gridDim.x;
  lp.gridDim.y = gridDim.y;
  lp.gridDim.z = gridDim.z;

  lp.groupDim.x = groupDim.x;
  lp.groupDim.y = groupDim.y;
  lp.groupDim.z = groupDim.z;

  lp.groupMemBytes = groupMemBytes;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp.av = stream ? &(stream->av) : &av;
  lp.cf = NULL;

  return lp;
}

hipError_t hipMalloc(void** ptr, size_t size) {
  *ptr = malloc(size);
  return hipSuccess;
}

hipError_t hipMallocHost(void** ptr, size_t size) {
  return hipMalloc(ptr, size);
}

// width in bytes
hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height) {
  if(width == 0 || height == 0) return hipErrorUnknown;
  *pitch = ((((int)width-1)/128) + 1)*128;
  *ptr = malloc((*pitch)*height);
  return hipSuccess;
}

// TODO: Improve this to include other things from desc
hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc,
                          size_t width, size_t height, unsigned int flags) {
  *array = (hipArray*)malloc(sizeof(hipArray));
  if(desc->f == hipChannelFormatKindFloat) {
    array[0]->data = (float*)malloc(width*height*sizeof(float));
  }
  array[0]->width = width;
  array[0]->height = height;
  return hipSuccess;
}

hipError_t hipMemcpy(void* dest, const void* src, size_t size, hipMemcpyKind kind) {
  // TODO: Does direction matter?
  memcpy(dest, src, size);
  return hipSuccess;
}

// dpitch, spitch, and width in bytes
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
  if(width > dpitch || width > spitch)
    return hipErrorUnknown;
// FIXME: generalize float
  int dp_sz = dpitch/sizeof(float);
  int sp_sz = spitch/sizeof(float);
  for(int i = 0; i < height; ++i) {
    memcpy((float*)dst + i*dp_sz, (float*)src + i*sp_sz, width);
  }
  return hipSuccess;
}

// wOffset, width, and spitch in bytes
hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
                                    size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
  if((wOffset + width > (dst->width * sizeof(float))) || width > spitch) {
    return hipErrorUnknown;
  }

// FIXME: generalize type
  int src_w = width/sizeof(float);
  int dst_w = dst->width;

  for(int i = 0; i < height; ++i) {
    memcpy((float*)dst->data + i*dst_w, (float*)src + i*src_w, width);
  }

  return hipSuccess;
}

//template <typename T, const int dim>
hipError_t hipBindTextureToArray(texture& tex, hipArray* array) {
  tex.data = array;
  return hipSuccess;
}

//template <typename T, const int dim>
hipError_t hipUnbindTexture(texture & tex) {
  tex.data = NULL;
  return hipSuccess;
}

hipError_t hipFree(void* ptr) {
  free(ptr);
  ptr = NULL;
  return hipSuccess;
}

hipError_t hipFreeHost(void* ptr) {
  return hipFree(ptr);
}

hipError_t hipMemset(void* ptr, int value, size_t count) {
  void * tmp = memset(ptr, value, count);
  return hipSuccess;
}


hipError_t hipStreamCreate(hipStream_t *stream) {
  *stream = new ihipStream_t((*hip_runtime::g_device)[hip_runtime::_the_device].create_view());
  // XXX(Yan-Ming): Error handling
  return hipSuccess;
}


hipError_t hipStreamSynchronize(hipStream_t stream) {
  if (stream == nullptr)
    hc::accelerator().get_default_view().wait();
  else
    stream->av.wait();
  return hipSuccess;
};


hipError_t hipStreamDestroy(hipStream_t stream) {
  delete stream;
  return hipSuccess;
}


hipError_t hipSetDevice(int device) {
  if (0 <= device && device < hip_runtime::g_device->size())
    hip_runtime::_the_device = device;
  return hipSuccess;
}


hipError_t hipGetDevice(int *device) {
  *device = hip_runtime::_the_device;
  return hipSuccess;
}


hipError_t hipGetDeviceCount(int *count) {
  *count = hip_runtime::g_device->size();
  return hipSuccess;
}

int hipDeviceSynchronize(void) {
  hc::accelerator().get_default_view().wait();
  return 0;
}

hipError_t hipMemcpyAsync(void *dst, const void *src,
                          size_t  count,
                          hipMemcpyKind kind,
                          hipStream_t stream) {
  // XXX(Yan-Ming): Does kind matter?
  char *d = (char *)dst;
  char *s = (char *)src;

  // byte by byte copy
  hc::parallel_for_each(stream ? stream->av :
                                 hc::accelerator().get_default_view(),
                        hc::extent<1>(count),
                        [s, d](hc::index<1> idx) __attribute((hc)) {
    d[idx[0]] = s[idx[0]];
  });

  return hipSuccess;
}


hipError_t hipMemsetAsync(void *dst, int value, size_t count,
                          hipStream_t stream) {
  char *d = (char *)dst;

  // byte by byte assignment
  hc::parallel_for_each(stream ? stream->av :
                                 hc::accelerator().get_default_view(),
                        hc::extent<1>(count),
                        [d, value](hc::index<1> idx) __attribute((hc)) {
    d[idx[0]] = value;
  });

  return hipSuccess;
}

// C++

hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f) {
  hipChannelFormatDesc cd;
  cd.x = x; cd.y = y; cd.z = z; cd.w = w;
  cd.f = f;
  return cd;
}

hipChannelFormatDesc hipCreateChannelDesc() {
  return hipCreateChannelDesc(0, 0, 0, 0, hipChannelFormatKindFloat);
}

uint3 DIM3(int x, int y, int z) {
  uint3 ret;
  ret.x = x;
  ret.y = y;
  ret.z = z;
  return ret;
}

uint3 DIM3(int x, int y) {
  return DIM3(x, y, 1);
}

uint3 DIM3(int x) {
  return DIM3(x, 1, 1);
}


#endif // #ifndef USE_CUDA
