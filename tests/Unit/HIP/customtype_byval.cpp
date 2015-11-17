// XFAIL:
// RUN: %hc %s -lhip_runtime -o %t.out && %t.out

#include "hip.h"
#include "hip_runtime.h"

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
  Bar* data2;

  data1.x = 5;
  hipMalloc((void**)&data2, SIZE*sizeof(Foo));

  dim3 grid = DIM3(GRID_SIZE, 1);
  dim3 block = DIM3(TILE_SIZE, 1);

  hipLaunchKernel(kernel1, grid, block, data1, data2);

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if((data2[i].x != i + data1.x)) {
      ret = 1;
      break;
    }
  }

 return ret;
}
