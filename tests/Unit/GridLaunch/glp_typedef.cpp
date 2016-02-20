// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "grid_launch.h"
#include "hc_am.hpp"

#define GRID_SIZE 16
#define TILE_SIZE 16

typedef grid_launch_parm newglp1;
typedef newglp1 newglp2;

__attribute__((hc_grid_launch)) void kernel1(const newglp1 lp, int* x) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  x[idx] = idx;
}

__attribute__((hc_grid_launch)) void kernel2(const newglp2 lp, int* x) {
  int idx = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  x[idx] = idx;
}

int main() {

  const int sz = GRID_SIZE*TILE_SIZE;

  int* data1 = (int* )malloc(sz*sizeof(int));
  int* data2 = (int* )malloc(sz*sizeof(int));

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(sz*sizeof(int), acc, 0);
  int* data2_d = (int*)hc::am_alloc(sz*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf1;
  lp.cf = &cf1;
  kernel1(lp, data1_d);
  lp.cf->wait();

  hc::completion_future cf2;
  lp.cf = &cf2;
  kernel2(lp, data2_d);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, sz*sizeof(int));
  hc::am_copy(data2, data2_d, sz*sizeof(int));

  bool ret = true;

  for(int i = 0; i < sz; ++i) {
    ret &= (data1[i] == i);
    ret &= (data2[i] == i);
  }

  hc::am_free(data1_d);
  hc::am_free(data2_d);
  free(data1);
  free(data2);

  return !ret;

}
