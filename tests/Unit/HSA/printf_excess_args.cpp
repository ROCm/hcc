// RUN: %hc %s -DHCC_ENABLE_ACCELERATOR_PRINTF -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <cassert>
#include <hc.hpp>
#include <hc_printf.hpp>

// create 2 tiles of 64 threads
#define TILE (64)
#define GLOBAL (TILE*2)

int main() {

  hc::parallel_for_each(hc::extent<1>(GLOBAL).tile(TILE), [=](hc::tiled_index<1> tidx) [[hc]] {

      // Passing 2 printf args with 1 specifier
      const char* str_extra1 = "GPU test A: %d\n";
      int extra1 = -1;
      hc::printf(str_extra1, extra1, tidx.global[0]);

      // Passing 3 printf args with 0 specifiers
      const char* str_extra2 = "GPU test B\n";
      int extra2 = -2;
      hc::printf(str_extra2, extra1, extra2, tidx.global[0]);


  }).wait();

  printf("GPU is done!\n");
  printf("CPU test C: %d\n", -1);

  return 0;
}


// CHECK-DAG: GPU test A: -1
// CHECK-DAG: GPU test B
// CHECK-DAG: GPU is done!
// CHECK-DAG: CPU test C: -1
