// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out | %FileCheck %s

#include <amp.h>
#include <hsa_printf.h>

#include <iostream>

#define SIZE (32)
#define PRINTF_BUFFER_SIZE (512)

int main() {
  HSAPrintfPacketQueue* q = createHSAPrintfPacketQueue(PRINTF_BUFFER_SIZE);
  
  using namespace concurrency;


  parallel_for_each(extent<1>(SIZE), [&](index<1> idx) restrict(amp) {
    const char* s1 = "Hello HSA from %s: %d\n";
    const char* s2 = "thread";
    const char* s6 = "Hello again from %s: %d\n";

    if (idx[0] == 0) {
      hsa_printf(q, s1, s2, idx[0]);
      hsa_printf(q, s6, s2, idx[0]);
    } else if (idx[0] == 5) {
      hsa_printf(q, s1, "work-item", idx[0]);
    } else if (idx[0] == 10) {
      hsa_printf(q, s1, "work item", idx[0]);
    } else if (idx[0] == 15) {
      hsa_printf(q, s1, "workitem", idx[0]);
      hsa_printf(q, s6, s2, idx[0]);
    }
  });

  dumpHSAPrintfPacketQueue(q);

  processHSAPrintfPacketQueue(q);

  destroyHSAPrintfPacketQueue(q);

  return 0;
}

// CHECK: Hello HSA from thread: 0
// CHECK: Hello again from thread: 0
// CHECK: Hello HSA from work-item: 5
// CHECK: Hello HSA from work item: 10
// CHECK: Hello HSA from workitem: 15
// CHECK: Hello again from thread: 15
