#pragma once

#include <stdint.h>

namespace hc{
class completion_future;
class accelerator_view;
}

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

  // customized serialization: don't need av and cf in kernel
/*  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize& s) const {
    s.Append(sizeof(int), &gridDim.x);
    s.Append(sizeof(int), &gridDim.y);
    s.Append(sizeof(int), &gridDim.z);
    s.Append(sizeof(int), &groupDim.x);
    s.Append(sizeof(int), &groupDim.y);
    s.Append(sizeof(int), &groupDim.z);
    s.Append(sizeof(int), &groupId.x);
    s.Append(sizeof(int), &groupId.y);
    s.Append(sizeof(int), &groupId.z);
    s.Append(sizeof(int), &threadId.x);
    s.Append(sizeof(int), &threadId.y);
    s.Append(sizeof(int), &threadId.z);
    s.Append(sizeof(unsigned), &groupMemBytes);
  }
*/
  __attribute__((annotate("user_deserialize")))
  grid_launch_parm(int gridDim_x,  int gridDim_y,  int gridDim_z,
                   int groupDim_x, int groupDim_y, int groupDim_z,
                   int groupId_x,  int groupId_y,  int groupId_z,
                   int threadId_x, int threadId_y, int threadId_z,
                   unsigned groupMemBytes_) {
    gridDim.x  = gridDim_x;
    gridDim.y  = gridDim_y;
    gridDim.z  = gridDim_z;
    groupDim.x = groupDim_x;
    groupDim.y = groupDim_y;
    groupDim.z = groupDim_z;
    groupId.x  = groupId_x;
    groupId.y  = groupId_y;
    groupId.z  = groupId_z;
    threadId.x = threadId_x;
    threadId.y = threadId_y;
    threadId.z = threadId_z;
    groupMemBytes = groupMemBytes_;
  }

} grid_launch_parm;
/*
// TODO: Will move to separate source file in the future
extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->gridDim.x = lp->gridDim.y = lp->gridDim.z = 1;

  lp->groupDim.x = lp->groupDim.y = lp->groupDim.z = 1;

  lp->groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
  lp->cf = NULL;
}
*/
