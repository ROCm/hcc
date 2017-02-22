
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "grid_launch.hpp"
#include "hc_am.hpp"
#include <iostream>

// Test for non-square tiles
#define TILE_I 16
#define TILE_J 8

__attribute__((hc_grid_launch)) void kernel_call(grid_launch_parm lp, int *a_d, int pitch)
{
  //int i = lp.group_id.x*TILE_I + lp.thread_id.x;
  //int j = lp.group_id.y*TILE_J + lp.thread_id.y;
  int i = amp_get_group_id(0) * TILE_I + amp_get_local_id(0);
  int j = amp_get_group_id(1) * TILE_J + amp_get_local_id(1);

  int i2d = i + j*pitch/sizeof(int);

  a_d[i2d] = i2d;
}

int main()
{
  int width = 320;
  int height = 112;

  int *a = (int*)malloc(width*height*sizeof(int));

  int pitch = sizeof(int)*width;

  auto acc = hc::accelerator();
  int* a_d = (int*)hc::am_alloc(width*height*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(width/TILE_I, height/TILE_J);
  lp.group_dim = gl_dim3(TILE_I, TILE_J);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel_call(lp, a_d, pitch);
  lp.cf->wait();

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(a_d, a, width*height*sizeof(int));

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

  hc::am_free(a_d);
  free(a);

  return 0;
}
