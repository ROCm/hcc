// XXX(Yan-Ming): borrow from AMS's hip repo
#pragma once

#include <hip.h>
#include <iostream>
#include <vector>

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

typedef enum hipError_t {
   hipSuccess = 0
} hipError_t;


static std::vector<hc::accelerator> *g_device = nullptr;
static int _the_device = 1;
void __attribute__ ((constructor)) hip_init() {
  g_device = new std::vector<hc::accelerator>(hc::accelerator().get_all());
  // use HSA device by default
  _the_device = 1;
}


inline hipError_t hipStreamCreate(hipStream_t *stream) {
  *stream = new ihipStream_t((*g_device)[_the_device].create_view());
  // XXX(Yan-Ming): Error handling
  return hipSuccess;
}


hipError_t hipStreamSynchronize(hipStream_t stream) {
  stream->av.wait();
  return hipSuccess;
};


hipError_t hipStreamDestroy(hipStream_t stream) {
  delete stream;
  return hipSuccess;
}


hipError_t hipSetDevice(int device) {
  if (0 <= device && device < g_device->size())
    _the_device = device;
  return hipSuccess;
}


hipError_t hipGetDevice(int *device) {
  *device = _the_device;
  return hipSuccess;
}


hipError_t hipGetDeviceCount(int *count) {
  *count = g_device->size();
  return hipSuccess;
}
