// RUN: %hc %s -DHCC_ENABLE_ACCELERATOR_PRINTF -DCHECK_PRINTF_ERROR -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <cassert>
#include <hc.hpp>
#include <hc_printf.hpp>

// create 2 tiles of 64 threads
#define TILE (64)
#define GLOBAL (TILE*2)

int main() {

#ifdef CHECK_PRINTF_ERROR 
  hc::array_view<hc::PrintfError,1> err(GLOBAL*2);
#endif

  hc::parallel_for_each(hc::extent<1>(GLOBAL).tile(TILE), [=](hc::tiled_index<1> tidx) [[hc]] {

      const char* str1 = "Hello HC from %s: %03d\n";
      const char* str2 = "thread";
      const char* str3 = "Hello again from %s: %03d\n";

#ifdef CHECK_PRINTF_ERROR 
      err[tidx.global[0]*2] = hc::printf(str1, str2, tidx.global[0]);
      err[tidx.global[0]*2 + 1] = hc::printf(str3, "thread", tidx.global[0]);
#else
      hc::printf(str1, str2, tidx.global[0]);
      hc::printf(str3, "thread", tidx.global[0]);
#endif

  }).wait();
  
#ifdef CHECK_PRINTF_ERROR 
  auto ex = err.get_extent();
  for (int i = 0; i < ex[0]; i++) {
    assert(err[i] == hc::PRINTF_SUCCESS);
  }
#endif

  return 0;
}

// CHECK-DAG: Hello HC from thread: 000
// CHECK-DAG: Hello HC from thread: 063
// CHECK-DAG: Hello HC from thread: 064
// CHECK-DAG: Hello HC from thread: 127
// CHECK-DAG: Hello again from thread: 000
// CHECK-DAG: Hello again from thread: 063
// CHECK-DAG: Hello again from thread: 064
// CHECK-DAG: Hello again from thread: 127
