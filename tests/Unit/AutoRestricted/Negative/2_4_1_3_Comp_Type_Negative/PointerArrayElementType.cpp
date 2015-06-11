// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
#include "common.h"

using std::vector;
using namespace concurrency;

//2_Cxx_Lang_Exte/2_4_amp_Rest_Modi/2_4_1_Rest_on_Type/2_4_1_3_Comp_Type/Negative/PointerArrayElementType/test.cpp
void PointerArrayElmentTypeNotSupported(int x) __AUTO {
  int * arr[5];
}

void AMP_AND_CPU_Func() restrict(cpu,amp) {
  PointerArrayElmentTypeNotSupported(1);
}
// CHECK: PointerArrayElementType.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: PointerArrayElmentTypeNotSupported(1);
// CHECK-NEXT: ^

int main(void)
{
  exit(1);
  return 0;
}

