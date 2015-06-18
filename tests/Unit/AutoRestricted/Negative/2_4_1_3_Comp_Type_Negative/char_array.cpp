// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
#include "common.h"

using std::vector;
using namespace concurrency;

// From 2_Cxx_Lang_Exte/2_4_amp_Rest_Modi/2_4_1_Rest_on_Type/2_4_1_3_Comp_Type/Negative/CharArray/test.cpp
void f_chararray() __AUTO
{
  char ca[10];
  ca[2] = 'c';
}

void AMP_AND_CPU_Func() restrict(cpu,amp) {
  f_chararray();
}
// CHECK: char_array.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_chararray();
// CHECK-NEXT: ^

int main(void)
{
  exit(1);
  return 0;
}

