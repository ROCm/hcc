// RUN: %hc %s -lhc_am -o %t.out && %t.out | %FileCheck %s

#include <hc.hpp>
#include <hc_printf.hpp>

#include <iostream>

#define TILE (16)
#define GLOBAL (TILE)

#define PRINTF_BUFFER_SIZE (256)

int main() {
  using namespace hc;

  accelerator acc = accelerator();
  PrintfPacket* printf_buf = createPrintfBuffer(acc, PRINTF_BUFFER_SIZE);

  if (!printf_buf) {
    std::printf("createPrintfBuffer failed.\n");
  }

  // Here we can print the string address with %p if we cast to (void*)
  parallel_for_each(extent<1>(GLOBAL).tile(TILE), [=](tiled_index<1> tidx) [[hc]] {
      const char* str1 = "Thread: %03d, String Address: %p\n";
      printf(printf_buf, str1, tidx.global[0], (void*)str1);
  }).wait();

  processPrintfBuffer(printf_buf);

  deletePrintfBuffer(printf_buf);

  return 0;
}

// CHECK-NOT: createPrintfBuffer failed.

// CHECK-DAG: Thread: 000, String Address: [[ADDR:0x[0-9a-f]+]]
// CHECK-DAG: Thread: 007, String Address: [[ADDR]]
// CHECK-DAG: Thread: 008, String Address: [[ADDR]]
// CHECK-DAG: Thread: 015, String Address: [[ADDR]]
