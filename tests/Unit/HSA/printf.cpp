// RUN: %hc %s -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <hc.hpp>
#include <hc_printf.hpp>

#include <iostream>

#define TILE (64)
#define GLOBAL (TILE*2)

int main() {
  using namespace hc;

  hc::array_view<hc::PrintfError,1> err(GLOBAL*2);

  parallel_for_each(extent<1>(GLOBAL).tile(TILE), [=](tiled_index<1> tidx) [[hc]] {
      const char* str1 = "Hello HC from %s: %03d\n";
      const char* str2 = "thread";
      const char* str3 = "Hello again from %s: %03d\n";
#if 1
      err[tidx.global[0]*2] = printf(str1, str2, tidx.global[0]);
      err[tidx.global[0]*2 + 1] = printf(str3, "thread", tidx.global[0]);
#else
      printf(str1, str2, tidx.global[0]);
      printf(str3, "thread", tidx.global[0]);
#endif
  }).wait();

  return 0;
}

// CHECK-NOT: createPrintfBuffer failed.

// CHECK-DAG: Hello HC from thread: 000
// CHECK-DAG: Hello HC from thread: 063
// CHECK-DAG: Hello HC from thread: 064
// CHECK-DAG: Hello HC from thread: 127
// CHECK-DAG: Hello again from thread: 000
// CHECK-DAG: Hello again from thread: 063
// CHECK-DAG: Hello again from thread: 064
// CHECK-DAG: Hello again from thread: 127
