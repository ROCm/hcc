// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "hc.hpp"
#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

#define GRID_SIZE 16
#define TILE_SIZE 16

// C++11 style attribute
[[hc]] int foo(grid_launch_parm lp) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  return idx;
}

// C-style attribute
__attribute__((hc)) int foo2(grid_launch_parm lp) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  return idx + 2;
}

// No attribute
int foo3(grid_launch_parm lp) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  return idx + 3;
}

__attribute__((hc_grid_launch)) void kernel(grid_launch_parm lp, int* x) {
  int idx = foo(lp);
  x[idx] = idx + foo2(lp) + foo3(lp);
}


int main() {

  const int sz = GRID_SIZE*TILE_SIZE;

  int* data1 = (int* )malloc(sz*sizeof(int));

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(sz*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel(lp, data1_d);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, sz*sizeof(int));

  bool ret = true;

  for(int i = 0; i < sz; ++i) {
    ret &= (data1[i] == i + i + 2 + i + 3);
  }

  hc::am_free(data1_d);
  free(data1);

  return !ret;

}
