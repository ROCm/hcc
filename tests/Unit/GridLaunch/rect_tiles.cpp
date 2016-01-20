// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

// Test for non-square tiles
#define TILE_I 16
#define TILE_J 8

__attribute__((hc_grid_launch)) void kernel_call(grid_launch_parm lp, int *a_d, int pitch)
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

  int pitch = sizeof(int)*width;

  int* a_d = (int*)hc::am_alloc(width*height*sizeof(int), hc::accelerator(), 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(width/TILE_I, height/TILE_J);
  lp.groupDim = gl_dim3(TILE_I, TILE_J);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel_call(lp, a_d, pitch);
  lp.cf->wait();

  hc::am_copy(a, a_d, width*height*sizeof(int));

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
