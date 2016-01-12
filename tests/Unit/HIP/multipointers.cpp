// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out

#include "hip.h"

#define HEIGHT 256
#define WIDTH 512
#define TILE_SIZE 16

#define SIZE WIDTH*HEIGHT;

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int **data2d) {
  int x = lp.threadId.x + lp.groupId.x*lp.groupDim.x;
  int y = lp.threadId.y + lp.groupId.y*lp.groupDim.y;

  data2d[y][x] = x + y*WIDTH;
}


int main(void) {

  int **data2d = (int **)malloc(HEIGHT*sizeof(int *));
  for(int j = 0; j < HEIGHT; ++j) {
    data2d[j] = (int *)malloc(WIDTH*sizeof(int));
  }

  dim3 grid = DIM3(WIDTH/TILE_SIZE, HEIGHT/TILE_SIZE);
  dim3 block = DIM3(TILE_SIZE, TILE_SIZE);

  hipLaunchKernel(kernel1, grid, block, data2d);

  bool ret = 0;
  for(int j = 0; j < HEIGHT; ++j) {
    for(int i = 0; i < WIDTH; ++i) {
      if(data2d[j][i] != i + j*WIDTH) {
          ret = 1;
          break;
      }
    }
  }

 return ret;
}
