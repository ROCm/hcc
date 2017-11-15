// RUN: %hc %s -E -o %t.i
// cat %t.gpu.i | %FileCheck %s

#include <cstdio>

int main() {

// CHECK-NOT: this_is_host_path
#ifdef __HCC_CPU__
  int this_is_host_path;
#endif

// CHECK: this_is_accelerator_path
#ifdef __HCC_ACCELERATOR__
  int this_is_accelerator_path;
#endif
  return 0;
}
