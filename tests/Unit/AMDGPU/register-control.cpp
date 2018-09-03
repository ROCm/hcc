// RUN: %hc %s -o %t.out -Xlinker -dump-llvm -Xlinker -dump-dir=%T %target_all_gpus
// RUN: %llvm-dis %T/dump-gfx803.opt.bc -f -o - | %FileCheck %s
// RUN: %t.out

#include <hc.hpp>
#include <vector>

#define GRID_SIZE (1024)

int main() {
  using namespace hc;
  array<unsigned int, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);
  // CHECK-LABEL: define weak_odr amdgpu_kernel void {{.*Kernel_emitter.*}}"
  // CHECK-SAME: {{[^#]*}}#[[ATTR0:[0-9]+]]
  auto k = make_callable_with_AMDGPU_attributes<
    Waves_per_eu<5, 6>,
    Flat_workgroup_size<1, 10>
    #if defined(NON_CLANG_ATTRIBUTES)
      , Max_workgroup_dim<10, 1, 1>
    #endif
    >([&](index<1>& idx) [[hc]] { table(idx) = idx[0]; }
  );
  parallel_for_each(ex, k).wait();

  // verify result
  bool ret = true;
  std::vector<unsigned int> result = table;
  for (int i = 0; i < GRID_SIZE; ++i) {
    ret &= (result[i] == i);
  }

  return !(ret == true);
}

// CHECK: attributes #[[ATTR0]] = {{{.*}}"amdgpu-flat-work-group-size"="1,10" "amdgpu-waves-per-eu"="5,6"