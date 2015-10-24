// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

// the test case would cause linker to complain about missing __cxa_thread_atexit

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <stdint.h>
#include <hc.hpp>


struct grid_launch_parm_s {
    uint32_t gridDim[3];
    uint32_t groupDim[3];

    uint32_t groupMemBytes;  // amount of group memory to reserve for each work-group.

    int av;   // Accelerator-view to target.

    bool    async;      // run kernel asynchronously (default)

    // Set only after launch:
    unsigned gridIdx[3];
    unsigned groupIdx[3];

    grid_launch_parm_s();
};

typedef struct grid_launch_parm_s grid_launch_parm;

// thread_local would cause linker to complain about missing __cxa_thread_atexit
thread_local hc::accelerator_view tls_accelerator_view = hc::accelerator().get_default_view();

#define hipThreadIdx_x (idx.local[0])
#define hipThreadIdx_y (idx.local[1])
#define hipThreadIdx_z (idx.local[2])

#define hipBlockIdx_x  (idx.tile[0])
#define hipBlockIdx_y  (idx.tile[1])
#define hipBlockIdx_z  (idx.tile[2])

#define hipBlockDim_x  (ext_tile.tile_dim[0])
#define hipBlockDim_y  (ext_tile.tile_dim[1])
#define hipBlockDim_z  (ext_tile.tile_dim[2])

#define hipGridDim_x  (ext_tile[0])
#define hipGridDim_y  (ext_tile[1])
#define hipGridDim_z  (ext_tile[2])


#define hipLaunchKernel(_kernelName, _numBlocks3D, _blockDim3D, _groupMemBytes, _streamId, ...) \
{\
  grid_launch_parm lp;\
  lp.gridDim[0] = _numBlocks3D.x * _blockDim3D.x;/*Convert from #blocks to #threads*/ \
  lp.gridDim[1] = _numBlocks3D.y * _blockDim3D.y;/*Convert from #blocks to #threads*/ \
  lp.gridDim[2] = _numBlocks3D.z * _blockDim3D.z;/*Convert from #blocks to #threads*/ \
  lp.groupDim[0] = _blockDim3D.x; \
  lp.groupDim[1] = _blockDim3D.y; \
  lp.groupDim[2] = _blockDim3D.z; \
  lp.groupMemBytes = _groupMemBytes;\
  lp.av = _streamId;\
  _kernelName(lp, __VA_ARGS__);\
}

typedef struct dim3 {
  uint32_t x;                 ///< x
  uint32_t y;                 ///< y
  uint32_t z;                 ///< z

  // C++ only, TODO -add compile gate.
  dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};

  // TODO - can we do some C++ magic to handle creation one of these on assignment from an int?  Sometimes users spec gridDim with one int.

} dim3;


#define MY_DEFAULT_AV 1 // TODO, hack, this could come from the Kalmar TLS default accelerator_view.

// C interface:
void grid_launch_init(grid_launch_parm *lp)
{
    lp->gridDim[0] = lp->gridDim[1] = lp->gridDim[2] = 1;
    lp->groupDim[0] = 256; lp->groupDim[1] = 1; lp->groupDim[2] = 1;

    lp->async = true;

    lp->groupMemBytes = 0;
    lp->av = MY_DEFAULT_AV; // TODO-GL TODO-Kalmar :

    lp->gridIdx[0]  = lp->gridIdx[1]  = lp->gridIdx[2]  = 0;
    lp->groupIdx[0] = lp->groupIdx[1] = lp->groupIdx[2] = 0;
}
grid_launch_parm_s::grid_launch_parm_s()
{
    grid_launch_init(this);
}


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

void vectoradd_float(const grid_launch_parm &lp, float* a, const float* b, const float* c, int width, int height, hc::extent<3>& ext, hc::tiled_extent<3>& ext_tile)
{
      ext_tile.set_dynamic_group_segment_size(lp.groupMemBytes);
    
      hc::completion_future cf = hc::parallel_for_each (
              tls_accelerator_view,
              ext_tile,
              [=, &ext_tile, &lp] (hc::tiled_index<3> idx) 
              __attribute__((hc))
  {

      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = b[i] + c[i];
      }



  }
      );
}



int main() {

  float* hostA;
  float* hostB;
  float* hostC;

  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }

  hc::extent<3> ext(WIDTH, HEIGHT, 1);
  auto ext_tile = ext.tile(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

  hipLaunchKernel(vectoradd_float,
                  //dim3(WIDTH, HEIGHT),
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  hostA ,hostB ,hostC ,WIDTH ,HEIGHT, ext, ext_tile);

  tls_accelerator_view.wait(); //this does work now

  // This also does work:
  hc::completion_future marker_cf = tls_accelerator_view.create_marker();
  marker_cf.wait();

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  free(hostA);
  free(hostB);
  free(hostC);

  return errors;
}
