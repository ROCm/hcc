// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <amp.h>
using namespace concurrency;

int f_dtor_mulitple() restrict(auto) {
  class MyClass
  { 
    public:
      MyClass() {}
      MyClass() restrict(amp) {}

      ~MyClass();
  };
  MyClass A;
}
// CHECK: Dtor_has_multiple_restrictions.cpp:[[@LINE-4]]:7: error: Destructor's restriction specifiers must cover the union of restrictions on all constructors
// CHECK-NEXT: ~MyClass();
// CHECK-NEXT: ^

int main(void)
{
  return 0;
}

