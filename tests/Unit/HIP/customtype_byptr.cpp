// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out

#include "grid_launch.h"

typedef struct {
  int x;
} Foo;

struct Bar {
  int x;
};

struct constStructconst {
  int x;
};

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, Foo *x, Bar *y, const constStructconst* C) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  x[i].x = i;
  y[i].x = i + C[i].x;
}


int main(void) {

  Foo* data1 = (Foo*)malloc(SIZE*sizeof(Foo));
  Bar* data2 = (Bar*)malloc(SIZE*sizeof(Bar));
  constStructconst* data3 = (constStructconst*)malloc(SIZE*sizeof(constStructconst));
  for(int i = 0; i < SIZE; ++i) {
    data3[i].x = i;
  }

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.gridDim = gl_dim3(GRID_SIZE, 1);
  lp.groupDim = gl_dim3(TILE_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1, data2, data3);
  lp.cf->wait();

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data1[i].x != i) || (data2[i].x != i + data3[i].x)) {
      ret = 1;
      break;
    }
  }

 return ret;
}
