// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <amp.h>
#include <hsa_printf.h>

#include <iostream>

#define SIZE (32)
#define PRINTF_BUFFER_SIZE (512)

int main() {
  HSAPrintfPacketQueue* q = createHSAPrintfPacketQueue(PRINTF_BUFFER_SIZE);
  
  using namespace concurrency;

  const char* s1 = "Hello HSA from %s: %d\n";
  const char* s2 = "thread";
  const char* s3 = "work-item";
  const char* s4 = "work item";
  const char* s5 = "workitem";
  const char* s6 = "Hello again from %s: %d\n";

  parallel_for_each(extent<1>(SIZE), [&](index<1> idx) restrict(amp) {
    if (idx[0] == 0) {
      hsa_printf(q, s1, s2, idx[0]);
      hsa_printf(q, s6, s2, idx[0]);
    } else if (idx[0] == 5) {
      hsa_printf(q, s1, s3, idx[0]);
    } else if (idx[0] == 10) {
      hsa_printf(q, s1, s4, idx[0]);
    } else if (idx[0] == 15) {
      hsa_printf(q, s1, s5, idx[0]);
      hsa_printf(q, s6, s2, idx[0]);
    }
  });

  dumpHSAPrintfPacketQueue(q);

  hsa_process_printf_queue(q);

  destroyHSAPrintfPacketQueue(q);

  return 0;
}

