// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

// Test passing a custom type by pointer

#include "hip.h"

#define WIDTH 64
#define HEIGHT 64
#define SIZE WIDTH*HEIGHT

#define GRID_SZ 16

__KERNEL void kernel_call(grid_launch_parm lp, float* data1, hipArray* array1)
{
  int idx = lp.threadId.x + lp.threadId.y*WIDTH;

  data1[idx] = array1->data[idx];

}

int main()
{

  hipArray* array1;

  float* data1;

  hipMalloc((void**)&data1, WIDTH*HEIGHT*sizeof(float));

  hipChannelFormatDesc desc = hipCreateChannelDesc();

  hipMallocArray(&array1, &desc, WIDTH, HEIGHT);
  for(int i = 0; i < SIZE; ++i)
    array1->data[i] = (float)i;

  dim3 grid = DIM3(WIDTH/GRID_SZ, HEIGHT/GRID_SZ);
  dim3 block = DIM3(GRID_SZ, GRID_SZ);

  hipLaunchKernel(kernel_call, grid, block, data1, array1);

  return 0;
}
