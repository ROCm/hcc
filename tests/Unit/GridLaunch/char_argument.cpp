// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "grid_launch.h"
#include "hc_am.hpp"

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int * data1, char c) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  data1[i] = (int)c;
}


int main(void) {

  char c = 1;

  int * data1 = (int*)malloc(SIZE*sizeof(int));
  auto acc = hc::accelerator();
  int * data1_d = (int*)hc::am_alloc(SIZE*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d, c);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, SIZE*sizeof(int));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data1[i] != (int)c)) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data1);
  free(data1);

  return ret;
}
