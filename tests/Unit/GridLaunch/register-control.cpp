// RUN: %hc -lhc_am %s -o %t.out -Xlinker -dump-llvm -Xlinker -dump-dir=%T
// RUN: %llvm-dis %T/dump*.opt.bc
// RUN: cat %T/dump*.opt.ll | %FileCheck %s
// RUN: %t.out

#include "grid_launch.hpp"
#include "hc_am.hpp"
#include "hc.hpp"
#include <iostream>

#define GRID_SIZE 256
#define TILE_SIZE 16

const int SIZE = GRID_SIZE*TILE_SIZE;

// CHECK-LABEL: define weak_odr amdgpu_kernel void @_ZN12_GLOBAL__N_138_Z7kernel116grid_launch_parmPi_functor19__cxxamp_trampolineEiiiiiiPi
// CHECK-SAME:({{[^)]*}}){{[^#]*}}#[[ATTR0:[0-9]+]]
// CHECK: attributes #[[ATTR0]] = {{{.*}}"amdgpu-flat-work-group-size"="1,10" "amdgpu-max-work-group-dim"="10,1,1" "amdgpu-waves-per-eu"="5,6"

__attribute__((hc_grid_launch)) void kernel1(grid_launch_parm lp, int *x)
[[hc_waves_per_eu(5,6)]]
[[hc_flat_workgroup_size(1,10)]]
[[hc_max_workgroup_dim(10,1,1)]]
{
  int i = hc_get_workitem_id(0) + hc_get_group_id(0)*lp.group_dim.x;

  x[i] = i;
}

int main(void) {

  int *data1 = (int *)malloc(SIZE*sizeof(int));

  auto acc = hc::accelerator();
  int* data1_d = (int*)hc::am_alloc(SIZE*sizeof(int), acc, 0);

  grid_launch_parm lp;
  grid_launch_init(&lp);

  lp.grid_dim.x = GRID_SIZE;
  lp.group_dim.x = TILE_SIZE;

  hc::completion_future cf;
  lp.cf = &cf;
  kernel1(lp, data1_d);
  lp.cf->wait();

  static hc::accelerator_view av = acc.get_default_view();
  av.copy(data1_d, data1, SIZE*sizeof(int));

  bool ret = 0;
  for(int i = 0; i < SIZE; ++i) {
    if(data1[i] != i) {
      ret = 1;
      break;
    }
  }

  hc::am_free(data1_d);
  free(data1);

  return ret;
}
