// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>

using namespace concurrency;

// before VC
int const_FUNC() restrict(auto) const {return 1;}
// CHECK: before_CV.cpp:[[@LINE-1]]:31: error: 'auto' restriction specifier is only allowed on function definition
// CHECK-NEXT:int const_FUNC() restrict(auto) const {return 1;}
// CHECK-NEXT:                              ^
// CHECK-NEXT:before_CV.cpp:[[@LINE-4]]:32: error: expected ';' after top level declarator
// CHECK-NEXT:int const_FUNC() restrict(auto) const {return 1;}
// CHECK-NEXT:                               ^
// CHECK-NEXT:                               ;
// CHECK-NEXT:before_CV.cpp:[[@LINE-8]]:39: error: expected unqualified-id
// CHECK-NEXT:int const_FUNC() restrict(auto) const {return 1;}

int main(void)
{
  return 0;
}

