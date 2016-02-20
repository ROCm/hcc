// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "hc.hpp"
#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

struct Foo {
  union {
    bool b[2];
    struct {
      int x;
      int y;
    };
  };
  size_t __padding1;
  size_t __padding2;
};

#define GRID_SIZE 16
#define TILE_SIZE 16

#define SIZE GRID_SIZE*TILE_SIZE

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int * data1, Foo F) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  data1[i] = i + F.x;
}


int main(void) {

  Foo F;
  F.x = 1337;
  F.y = 1336;

  F.b[0] = 1;
  F.b[1] = 2;

  int* data1 = (int*)malloc(SIZE*sizeof(int));
  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(SIZE*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d, F);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, SIZE*sizeof(int));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data1[i] != i + F.x)) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data1_d);
  free(data1);

  return ret;
}
