// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

typedef struct Foo {
  // dummy members so llvm won't coerce struct
  int d0;
  int d1;
  int d2;
  int d3;

  char x;
} Foo;

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int * data1, Foo f) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  data1[i] = f.x;
}


int main(void) {

  Foo f;
  f.x = 1;

  int * data1 = (int*)malloc(SIZE*sizeof(int));
  int * data1_d = (int*)hc::am_alloc(SIZE*sizeof(int), hc::accelerator(), 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d, f);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, SIZE*sizeof(int));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data1[i] != f.x)) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data1);
  free(data1);

  return ret;
}
