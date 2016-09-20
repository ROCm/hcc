
// RUN: %hc -lhc_am %s -o %t.out && %t.out

// FIXME: GridLaunch tests would hang HSA dGPU if executed in multi-thread
// environment. Need further invetigation


#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"
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
  [[hc]][[cpu]] int getX() { return x;}
  [[hc]][[cpu]] int getY() { return y;}
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
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;

  y[i].x = i + x.getY();
}


int main(void) {

  Foo data1(5);
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Bar));

  auto acc = hc::accelerator();
  Bar* data2_d = (Bar*)hc::am_alloc(SIZE*sizeof(Bar), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(GRID_SIZE, 1);
  lp.group_dim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1, data2_d);
  lp.cf->wait();

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data2_d, data2, SIZE*sizeof(Bar));

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
