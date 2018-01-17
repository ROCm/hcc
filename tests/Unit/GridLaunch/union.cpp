
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "hc.hpp"
#include "grid_launch.hpp"
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
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;


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

  lp.grid_dim = gl_dim3(GRID_SIZE, 1);
  lp.group_dim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d, F);
  lp.cf->wait();

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data1_d, data1, SIZE*sizeof(int));

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
