// RUN: %hc %s -DHCC_ENABLE_ACCELERATOR_PRINTF -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <hc.hpp>
#include <hc_printf.hpp>

#include <iostream>

#define GLOBAL (64)

int main() {

  // Here we can print the string address with %p if we cast to (void*)
  hc::parallel_for_each(hc::extent<1>(GLOBAL), [=](hc::index<1> idx) [[hc]] {
      const char* str1 = "Thread: %03d, String Address: %p\n";
      hc::printf(str1, idx[0], (void*)str1);
  }).wait();

  return 0;
}


// CHECK-DAG: Thread: 000, String Address: [[ADDR:0x[0-9a-f]+]]
// CHECK-DAG: Thread: 007, String Address: [[ADDR]]
// CHECK-DAG: Thread: 008, String Address: [[ADDR]]
// CHECK-DAG: Thread: 015, String Address: [[ADDR]]
