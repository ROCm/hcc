// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <cmath>
#include <hc.hpp>
#include <hip.h>

#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16

using namespace hc;

__KERNEL void scalarMulAdd(grid_launch_parm lp, float* out, const float *in, const float scalar, int width, int height)
{

  GRID_LAUNCH_INIT(lp);

  int x = lp.groupDim.x * lp.groupId.x + lp.threadId.x;
  int y = lp.groupDim.y * lp.groupId.y + lp.threadId.y;

  int i = y * width + x;
  if ( i < (width * height)) {
    out[i] += in[i] * scalar;
  }
}

int main() {

  float* A = new float[NUM];
  float* B = new float[NUM];
  float* C = new float[NUM];

  float* hostD = (float*)malloc(NUM * sizeof(float));
  float* deviceD = NULL;
  HC_ASSERT(hcMalloc((void**)&deviceD, NUM * sizeof(float)));

  // initialize the input data
  for (int i = 0; i < NUM; i++) {
    B[i] = (float)i;
    C[i] = (float)i*100.0f;
    hostD[i] = (float)i*321.0f;
  }

  // launch kernel
  parallel_for_each(
    extent<2>(WIDTH, HEIGHT).tile(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
    [=](tiled_index<2>& idx) __attribute((hc))
    {
      int x = idx.tile_dim[0] * idx.tile[0] + idx.local[0];
      int y = idx.tile_dim[1] * idx.tile[1] + idx.local[1];

      int i = y * WIDTH + x;
      if (i < NUM) {
        A[i] = B[i] + C[i];
      }
  }).wait();

  HC_ASSERT(hcMemcpy(deviceD, hostD, NUM*sizeof(float), hcMemcpyHostToAccelerator));
  const float scalar = 77;

  grid_launch_parm lp = hcCreateLaunchParam2(
    DIM3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
    DIM3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y));
  scalarMulAdd(lp, A, deviceD, scalar, WIDTH, HEIGHT);

  HC_ASSERT(hcMemcpy(hostD, deviceD, NUM*sizeof(float), hcMemcpyAcceleratorToHost));


  // verify the results
  int errors = 0;
  for (int i = 0; i < NUM; i++) {
    float expect = B[i] + C[i];
    expect += hostD[i] * scalar;
    float actual = A[i];
    if(std::abs(actual - expect)/((actual + expect)/2) > 0.00001) {
      errors++;
    }
  }
  if (errors!=0) {
    std::cout << errors << " errors" << std::endl;
  }

  delete [] A;
  delete [] B;
  delete [] C;

  return errors;
}

// CHECK:
