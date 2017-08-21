// RUN: %hc %s -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <hc.hpp>
#include <hc_printf.hpp>

#include <iostream>

#define TILE (64)
#define GLOBAL (TILE*2)

#define PRINTF_BUFFER_SIZE (16384)

int main() {
  using namespace hc;

  accelerator acc = accelerator();
  char* printf_fbuf = NULL;
  PrintfPacket* printf_buf = createPrintfBuffer(acc, PRINTF_BUFFER_SIZE, printf_fbuf);

  parallel_for_each(extent<1>(GLOBAL).tile(TILE), [=](tiled_index<1> tidx) [[hc]] {
      const char* str1 = "Hello HC from %s: %03d\n";
      const char* str2 = "thread";
      const char* str3 = "Hello again from %s: %03d\n";
      printf(printf_buf, printf_fbuf, str1, str2, tidx.global[0]);
      printf(printf_buf, printf_fbuf, str3, "thread", tidx.global[0]);
  }).wait();

  processPrintfBuffer(printf_buf, printf_fbuf);

  deletePrintfBuffer(printf_buf, printf_fbuf);

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
