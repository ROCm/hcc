// XFAIL: Linux
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "grid_launch.h"
#include "hc_am.hpp"
#include <iostream>

class Foo2 {
  int x;
};

class Foo {
public:
  Foo(int _x) : x(0), z(0), a(0), b(0) {y = _x; };
  int x;
  int y;
  int z;
  int a;
  int b;
  Foo2 foo2;
  int getX() { return x;}
  int getY() { return y;}
};

struct Bar {
  int x;
  int y;
  int z;
};

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, Foo x, Bar *y) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  y[i].x = i + x.getY();
}


int main(void) {

  Foo data1(5);
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Bar));

  auto acc = hc::accelerator();
  Bar* data2_d = (Bar*)hc::am_alloc(SIZE*sizeof(Bar), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1, data2_d);
  lp.cf->wait();

  hc::am_copy(data2, data2_d, SIZE*sizeof(Bar));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data2[i].x != i + data1.y)) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data2_d);
  free(data2);

  return ret;
}
