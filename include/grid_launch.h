#pragma once

#include <stdint.h>

#include <hc.hpp>

typedef struct uint3
{
  int x,y,z;
  uint3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
} uint3;

typedef struct grid_launch_parm
{
  uint3      gridDim;
  uint3      groupDim;
  uint3      groupId;
  uint3      threadId;
  unsigned int  groupMemBytes;
  // use acc_view for PFE in WrapperGen
  hc::accelerator_view  *av;
  hc::completion_future *cf;
} grid_launch_parm;

// TODO: Will move to separate source file in the future
extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->gridDim.x = lp->gridDim.y = lp->gridDim.z = 1;

  lp->groupDim.x = lp->groupDim.y = lp->groupDim.z = 1;

  lp->groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
  lp->cf = NULL;
}
