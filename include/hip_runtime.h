// XXX(Yan-Ming): borrow from AMS's hip repo
#pragma once

#include <hip.h>
#include <iostream>

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


void __attribute__ ((constructor)) hip_init() {
  // XXX(Yan-Ming): initialize global resource here
}


inline hipError_t hipStreamCreate(hipStream_t *stream) {
  *stream = new ihipStream_t(hc::accelerator().create_view());
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
