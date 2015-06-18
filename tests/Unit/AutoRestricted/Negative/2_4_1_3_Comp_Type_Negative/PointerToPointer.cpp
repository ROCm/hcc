// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
#include "common.h"

using std::vector;
using namespace concurrency;

void PointerToPointerNotSupported(int x) __AUTO {
  int ** ptr;
  return;
}

void AMP_AND_CPU_Func() restrict(cpu,amp) {
  PointerToPointerNotSupported(1);
}
// CHECK: PointerToPointer.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: PointerToPointerNotSupported(1);
// CHECK-NEXT: ^

int main(void)
{
  exit(1);
  return 0;
}

