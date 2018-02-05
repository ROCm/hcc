
// RUN: %hc -lhc_am %s -o %t.out && %t.out

#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"
#include <iostream>

#define GRID_SIZE 256

// This test case checks whether it's possible to pass short types as kernel
// arguments. On HCC runtime which doesn't support it, the test case would hang
// HSA runtime.

// seed kernel argument would hang HCC runtime in case it doesn't handle short
// type as a possible kernel argument type
__attribute__((hc_grid_launch)) void kernel(grid_launch_parm lp, short seed, short * g_data) {
  int idx = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;
  short tmps = g_data[idx];
  if (tmps == (short)-1)
    g_data[idx] = tmps;
}


int main(void) {

  double * data1 = (double*)malloc(GRID_SIZE*sizeof(double));
  memset(data1, 0, sizeof(GRID_SIZE*sizeof(double)));
  auto acc = hc::accelerator();
  double * data1_d = (double*)hc::am_alloc(GRID_SIZE*sizeof(double), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim = gl_dim3(1, 1);
  lp.group_dim = gl_dim3(GRID_SIZE, 1);

  hc::completion_future cf;
  lp.cf = &cf;
  kernel(lp, (short)1, (short*)data1_d);
  lp.cf->wait();

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data1_d, data1, GRID_SIZE*sizeof(double));

  hc::am_free(data1);
  free(data1);

  bool ret = 0;

  return ret;
}
