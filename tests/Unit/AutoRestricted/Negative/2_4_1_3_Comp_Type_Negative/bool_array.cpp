// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
#include "common.h"

using std::vector;
using namespace concurrency;

void BoolNotAllowedAsArrayElementType(int x) restrict(auto)
{
  bool arr[5]; // expected error{{bool is not allowed element type of array in amp restricted code}}
}

void AMP_AND_CPU_Func() restrict(cpu,amp) {
  BoolNotAllowedAsArrayElementType(1);
}
// CHECK: bool_array.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: BoolNotAllowedAsArrayElementType(1);
// CHECK-NEXT: ^

int main(void)
{
  exit(1);
  return 0;
}

