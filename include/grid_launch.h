#pragma once

#include <stdint.h>
#include <hc.hpp>

// Grid launch attributes for clang
#define __KERNEL __attribute__((hc_grid_launch))

#define __GROUP static __attribute__((address_space(3)))
#define __syncthreads() hc_barrier(CLK_LOCAL_MEM_FENCE)

// Prevent host-side compilation from compiler errors
#ifndef __GPU__
#define hc_barrier(n)
#endif

typedef struct
{
  int x,y,z;
} uint3;

typedef struct grid_launch_parm
{
  uint3      gridDim;
  uint3      groupDim;
  uint3      groupId;
  uint3      threadId;
  unsigned int  groupMemBytes;
#ifndef USE_CUDA
  // use acc_view for PFE in WrapperGen
  hc::accelerator_view  *av;
#endif
} grid_launch_parm;


