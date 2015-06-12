// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

//restriction after throw
void f_after_throw() throw(...) restrct(auto)
{}
// CHECK: after_throw_keyword_1.cpp:[[@LINE-2]]:33: error: expected function body after function declarator
// CHECK-NEXT:void f_after_throw() throw(...) restrct(auto)
// CHECK-NEXT:                                ^

void AMP_AND_CPU_Func() restrict(cpu,amp) {
  f_throw();
}

int main(void)
{
  return 0;
}

