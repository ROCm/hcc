#pragma once

#include "grid_launch.h"
#include "hc.hpp"

class grid_launch_parm_cxx : public grid_launch_parm
{
public:
  grid_launch_parm_cxx() = default;

};

extern inline void grid_launch_init(grid_launch_parm *lp) {
  lp->gridDim.x = lp->gridDim.y = lp->gridDim.z = 1;

  lp->groupDim.x = lp->groupDim.y = lp->groupDim.z = 1;

  lp->groupMemBytes = 0;
  static hc::accelerator_view av = hc::accelerator().get_default_view();
  lp->av = &av;
  lp->cf = NULL;
}

