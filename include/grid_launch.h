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

// TODO: Will move to separate source file in the future
extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->gridDim.x = lp->gridDim.y = lp->gridDim.z = 1;

  lp->groupDim.x = lp->groupDim.y = lp->groupDim.z = 1;

  lp->groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
}
