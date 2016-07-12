// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "grid_launch.hpp"
#include "hc_am.hpp"
#include <iostream>

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int *x) {
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;


  x[i] = i;
}

int main(void) {

  int *data1 = (int *)malloc(SIZE*sizeof(int));

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(SIZE*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim.x = GRID_SIZE;
  lp.group_dim.x = TILE_SIZE;

  hc::completion_future cf;
  lp.cf = &cf;
  // Test if lp.av is set to NULL
  lp.av = NULL;
  kernel1(lp, data1_d);
  lp.cf->wait();

  hc::am_copy(data1, data1_d, SIZE*sizeof(int));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if(data1[i] != i) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data1_d);
  free(data1);

  return ret;
}
