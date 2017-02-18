// RUN: %hc %s -o %t.out -Xlinker -dump-llvm -Xlinker -dump-dir=%T
// RUN: %llvm-dis %T/dump*.opt.bc
// RUN: cat %T/dump*.opt.ll| %FileCheck %s
// RUN: %t.out

#include <hc.hpp>
#include <vector>

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<unsigned int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  // CHECK-LABEL: define weak_odr amdgpu_kernel void @"_ZZ4mainEN3$_019__cxxamp_trampolineEPjii"
  // CHECK-SAME:({{[^)]*}}){{[^#]*}}#[[ATTR0:[0-9]+]]
  // CHECK: attributes #[[ATTR0]] = {{{.*}}"amdgpu-flat-work-group-size"="1,10" "amdgpu-max-work-group-dim"="10,1,1" "amdgpu-waves-per-eu"="5,6"
  auto k = [&](index<1>& idx) [[hc]]
                              [[hc_waves_per_eu(5,6)]]
                              [[hc_flat_workgroup_size(1,10)]]
                              [[hc_max_workgroup_dim(10,1,1)]]{
    table(idx) = idx[0];
  };
  parallel_for_each(ex, k ).wait();

  // verify result
  bool ret = true;
  std::vector<unsigned int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result[i] == i);
  }

  return !(ret == true);
}

