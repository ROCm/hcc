#pragma once

#include <stdint.h>

#include <hc_defines.h>

namespace hc{
class completion_future;
class accelerator_view;
}

namespace Kalmar {
class Serialize;
};


typedef struct gl_dim3
{
  int x,y,z;
  gl_dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
} gl_dim3;

typedef struct grid_launch_parm
{
  gl_dim3      gridDim;
  gl_dim3      groupDim;
  gl_dim3      groupId;
  gl_dim3      threadId;
  unsigned int  groupMemBytes;
  // use acc_view for PFE in WrapperGen
  hc::accelerator_view  *av;
  hc::completion_future *cf;

  grid_launch_parm() = default;

} grid_launch_parm;

