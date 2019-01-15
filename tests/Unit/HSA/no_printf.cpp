// RUN: %hc %s  -o %t.out && %t.out | %FileCheck -allow-empty %s

#include <hc/hc.hpp>
#include <hc_printf.hpp>

int main() {
  hc::parallel_for_each(hc::extent<1>(1), [](hc::index<1>) [[hc]] {
      hc::printf("Accelerator: Hello World!\n");
  }).wait();
  return 0;
}

// CHECK-NOT: Accelerator: Hello World!
