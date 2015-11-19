// XFAIL: Linux
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

#include <hip_runtime.h>

// Test for non-square tiles
#define TILE_I 16
#define TILE_J 8

__KERNEL void kernel_call(grid_launch_parm lp, int *a_d, int pitch)
{
  int i = lp.groupId.x*TILE_I + lp.threadId.x;
  int j = lp.groupId.y*TILE_J + lp.threadId.y;

  int i2d = i + j*pitch/sizeof(int);

  a_d[i2d] = i2d;
}

int main()
{
  int width = 320;
  int height = 112;

  int *a = (int*)malloc(width*height*sizeof(int));
  int *a_d;

  int pitch = sizeof(int)*width;
  hipMalloc((void**)&a_d, sizeof(int)*width*height);

  dim3 grid = dim3(width/TILE_I, height/TILE_J);
  dim3 block = dim3(TILE_I, TILE_J);

  hipLaunchKernel(kernel_call, grid, block, 0, 0, a_d, pitch);

  hipMemcpy((void *)a, (void *)a_d, width*sizeof(int)*height, hipMemcpyDeviceToHost);

  int ret = 0;
  for(int i = 0; i < width*height; ++i)
  {
    if(a[i] != i)
      ret++;
  }
  if(ret != 0)
  {
    printf("errors: %d\n", ret);
    return 1;
  }
  return 0;
}
