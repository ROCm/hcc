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


grid_launch_parm hipCreateLaunchParam2(hc_uint3 gridDim, hc_uint3 groupDim) {
  grid_launch_parm lp;

  lp.gridDim.x = gridDim.x;
  lp.gridDim.y = gridDim.y;
  lp.gridDim.z = gridDim.z;

  lp.groupDim.x = groupDim.x;
  lp.groupDim.y = groupDim.y;
  lp.groupDim.z = groupDim.z;

  lp.groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp.av = &av;

  return lp;
}


grid_launch_parm hipCreateLaunchParam3(hc_uint3 gridDim, hc_uint3 groupDim,
                                       int groupMemBytes) {
  grid_launch_parm lp;

  lp.gridDim.x = gridDim.x;
  lp.gridDim.y = gridDim.y;
  lp.gridDim.z = gridDim.z;

  lp.groupDim.x = groupDim.x;
  lp.groupDim.y = groupDim.y;
  lp.groupDim.z = groupDim.z;

  lp.groupMemBytes = groupMemBytes;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp.av = &av;

  return lp;
}


grid_launch_parm hipCreateLaunchParam4(hc_uint3 gridDim, hc_uint3 groupDim,
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

  return lp;
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
#endif // #ifndef USE_CUDA
