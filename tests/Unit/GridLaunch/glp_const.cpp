// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "hc.hpp"
#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

#define GRID_SIZE 16
#define TILE_SIZE 16

__attribute__((hc_grid_launch)) void kernel(const grid_launch_parm lp, int* x) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  x[idx] = idx;
}


int main() {

  const int sz = 10;

  int* data1 = (int* )malloc(sz*sizeof(int));

  int* data1_d = (int*)hc::am_alloc(sz*sizeof(int), hc::accelerator(), 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = uint3(GRID_SIZE, 1);
  lp.groupDim = uint3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel(lp, data1_d);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, sz*sizeof(int));

  bool ret = true;

  for(int i = 0; i < sz; ++i) {
    ret &= (data1[i] == i);
  }

  return !ret;

}
