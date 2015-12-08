// XFAIL: Linux
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

// Test launching three kernels via two different streams

#include "hip_runtime.h"

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__KERNEL void kernel1(grid_launch_parm lp, int* data1) {
  int idx = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;

  data1[idx] += 1;
}

__KERNEL void kernel2(grid_launch_parm lp, int* data2) {
  int idx = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;

  data2[idx] += 2;
}

__KERNEL void kernel3(grid_launch_parm lp, int* data3) {
  int idx = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;

  data3[idx] += 3;
}

int main(void) {

  hipStream_t stream1;
  hipStream_t stream2;

  hipStreamCreate(&stream1);
  hipStreamCreate(&stream2);

  int* data1;
  int* data2;
  int* data3;

  hipMallocHost((void**)&data1, SIZE*sizeof(int));
  hipMallocHost((void**)&data2, SIZE*sizeof(int));
  hipMallocHost((void**)&data3, SIZE*sizeof(int));

  for(int i = 0; i < SIZE; ++i)
    data3[i] = data2[i] = data1[i] = i;

  dim3 grid = dim3(GRID_SIZE, 1);
  dim3 block = dim3(TILE_SIZE, 1);

  hipLaunchKernel(kernel1, grid, block, 0, stream1, data1);
  hipLaunchKernel(kernel2, grid, block, 0, stream2, data2);
  hipLaunchKernel(kernel3, grid, block, 0, stream1, data3);

  hipStreamSynchronize(stream1);
  hipStreamSynchronize(stream2);

  bool ret = true;
  for(int i = 0; i < SIZE; ++i) {
    ret &= (data1[i] - i) == 1;
    ret &= (data2[i] - i) == 2;
    ret &= (data3[i] - i) == 3;
  }

  hipStreamDestroy(stream1);
  hipStreamDestroy(stream2);

  return !ret;
}
