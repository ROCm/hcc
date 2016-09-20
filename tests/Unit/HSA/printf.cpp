
// RUN: %hc %s -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <hc.hpp>
#include <hc_printf.hpp>

#include <iostream>

#define SIZE (32)
#define PRINTF_BUFFER_SIZE (512)

int main() {
  using namespace hc;

  accelerator acc = accelerator();
  PrintfPacket* printf_buf = createPrintfBuffer(acc, PRINTF_BUFFER_SIZE);

  const char* str1 = "Hello HC from %s: %d\n";
  const char* str2 = "thread";
  const char* str3 = "Hello again from %s: %d\n";

  parallel_for_each(extent<1>(SIZE), [=](index<1> idx) restrict(amp) {

    if (idx[0] == 0) {
      printf(printf_buf, str1, str2, idx[0]);
      printf(printf_buf, str3, str2, idx[0]);
    }

#if 0
    /* TBD.
       On dGPU, device addresses of strings on device global memory
       need to be fetched by hcc runtime somehow */
    } else if (idx[0] == 5) {
      printf(printf_buf, s1, "work-item", idx[0]);
    } else if (idx[0] == 10) {
      printf(printf_buf, s1, "work item", idx[0]);
    } else if (idx[0] == 15) {
      printf(printf_buf, s1, "workitem", idx[0]);
      printf(printf_buf, "Hello again from %s: %d\n", "thread", idx[0]);
    }
#endif

  }).wait();

  processPrintfBuffer(printf_buf);

  deletePrintfBuffer(printf_buf);

  return 0;
}

// CHECK: Hello HC from thread: 0
// CHECK: Hello again from thread: 0
