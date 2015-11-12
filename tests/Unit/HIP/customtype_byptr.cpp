// XFAIL: Linux
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

#include "hip.h"
#include "hip_runtime.h"

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

__KERNEL void kernel1(grid_launch_parm lp, Foo *x, Bar *y, const constStructconst* C) {
  int i = lp.threadId.x + lp.groupId.x*lp.groupDim.x;

  x[i].x = i;
  y[i].x = i + C[i].x;
}


int main(void) {

  Foo* data1;
  Bar* data2;
  constStructconst* data3;

  hipMalloc((void**)&data1, SIZE*sizeof(Foo));
  hipMalloc((void**)&data2, SIZE*sizeof(Bar));
  hipMalloc((void**)&data3, SIZE*sizeof(constStructconst));
  for(int i = 0; i < SIZE; ++i) {
    data3[i].x = i;
  }

  dim3 grid = DIM3(GRID_SIZE, 1);
  dim3 block = DIM3(TILE_SIZE, 1);

  hipLaunchKernel(kernel1, grid, block, data1, data2, data3);

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data1[i].x != i) || (data2[i].x != i + data3[i].x)) {
      ret = 1;
      break;
    }
  }

 return ret;
}
