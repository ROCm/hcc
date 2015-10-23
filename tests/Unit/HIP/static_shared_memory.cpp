// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hip.h>

// Test static shared memory in kernel

#define GRID_SIZE 256
#define TILE_SIZE 16

// Simple rotate inside tiles only
__KERNEL void staticSharedMemory(grid_launch_parm lp, int* in_data)
{
  __GROUP int shared[TILE_SIZE];

  int global = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  shared[lp.threadId.x] = in_data[global];

  __syncthreads();

  in_data[global] = shared[(lp.threadId.x + 1) % TILE_SIZE];
}

int main()
{
  const int array_size = GRID_SIZE*TILE_SIZE;
  int* in = (int*)malloc(array_size*sizeof(int));
  for(int i = 0; i < array_size; ++i)
  {
    in[i] = i;
  }
  int* in_data;
  hipMalloc((void**)&in_data, array_size*sizeof(int));
  hipMemcpy(in_data, in, array_size*sizeof(int), hipMemcpyHostToDevice);

  hipLaunchKernel(staticSharedMemory, DIM3(GRID_SIZE,1), DIM3(TILE_SIZE,1), in_data);

  hipMemcpy(in, in_data, array_size*sizeof(int), hipMemcpyDeviceToHost);

  int ret = 0;
  for(int i = 0; i < array_size; ++i)
  {
    if((i % TILE_SIZE) == (TILE_SIZE - 1))
    {
      if(in[i] != i - (TILE_SIZE - 1))
      {
        ret = 1;
        break;
      }
    }
    else if(in[i] != (i + 1))
    {
      ret = 1;
      break;
    }
  }
  return ret;
}
