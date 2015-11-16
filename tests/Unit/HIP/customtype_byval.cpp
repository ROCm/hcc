// XFAIL:
// RUN: %hc %s -o %t.out && %t.out

#include "grid_launch.h"

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

__KERNEL void kernel1(grid_launch_parm lp, Foo x, Bar *y) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  y[i].x = i + x.getY();
}


int main(void) {

  Foo data1(5);
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Foo));

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = uint3(GRID_SIZE, 1);
  lp.groupDim = uint3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1, data2);
  lp.cf->wait();

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data2[i].x != i + data1.y)) {
      ret = 1;
      break;
    }
  }

 return ret;
}
