#pragma once

#include "grid_launch.h"
#include "hc.hpp"

class grid_launch_parm_cxx : public grid_launch_parm
{
public:
  grid_launch_parm_cxx() = default;

  // customized serialization: don't need av and cf in kernel
  __attribute__((annotate("serialize")))
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

  __attribute__((annotate("user_deserialize")))
  grid_launch_parm_cxx(int gridDim_x,  int gridDim_y,  int gridDim_z,
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

};

extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->gridDim.x = lp->gridDim.y = lp->gridDim.z = 1;

  lp->groupDim.x = lp->groupDim.y = lp->groupDim.z = 1;

  lp->groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
  lp->cf = NULL;
}

