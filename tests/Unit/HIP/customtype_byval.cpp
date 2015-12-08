// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include "grid_launch.h"

typedef struct {
  int x;
} Foo;

struct Bar {
  int x;
};

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__KERNEL void kernel1(grid_launch_parm lp, Foo x, Bar *y) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  y[i].x = i + x.x;
}


int main(void) {

  Foo data1;
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Foo));

  data1.x = 5;

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
    if((data2[i].x != i + data1.x)) {
      ret = 1;
      break;
    }
  }

 return ret;
}
