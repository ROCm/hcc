
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"
#include <iostream>

typedef struct {
  int x;
} Foo;

struct Bar {
  int x;
};

namespace {
  struct constStructconst {
    int x;
  };
}

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, Foo *x, Bar *y, const constStructconst* C) {
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;

  x[i].x = i;
  y[i].x = i + C[i].x;
}


int main(void) {

  Foo* data1 = (Foo*)malloc(SIZE*sizeof(Foo));
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Bar));
  constStructconst* data3 = (constStructconst*)malloc(SIZE*sizeof(constStructconst));
  for(int i = 0; i < SIZE; ++i) 
    data3[i].x = i;

  auto acc = hc::accelerator();
  Foo* data1_d = (Foo*)hc::am_alloc(SIZE*sizeof(Foo), acc, 0);
  Bar* data2_d = (Bar*)hc::am_alloc(SIZE*sizeof(Bar), acc, 0);
  constStructconst* data3_d = (constStructconst*)hc::am_alloc(SIZE*sizeof(constStructconst), acc, 0);

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data3, data3_d, SIZE*sizeof(constStructconst));

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(GRID_SIZE, 1);
  lp.group_dim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d, data2_d, data3_d);
  lp.cf->wait();

  av.copy(data1_d, data1, SIZE*sizeof(Foo));
  av.copy(data2_d, data2, SIZE*sizeof(Bar));

  bool ret = 0;
  for (int i = 0; i < SIZE; i++)
  {
    if(data1[i].x != (data2[i].x - data3[i].x))
      ret = 1;
  }

  hc::am_free(data2_d);
  hc::am_free(data3_d);
  free(data1);
  free(data2);
  free(data3);

  return ret;
}
