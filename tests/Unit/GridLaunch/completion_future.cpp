// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out

#include "grid_launch.h"

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int *x) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  x[i] = i;
}

int main(void) {

  int *data1 = (int *)malloc(SIZE*sizeof(int));

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim.x = GRID_SIZE;
  lp.groupDim.x = TILE_SIZE;

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1);
  lp.cf->wait();

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if(data1[i] != i) {
      ret = 1;
      break;
    }
  }

 return ret;
}
