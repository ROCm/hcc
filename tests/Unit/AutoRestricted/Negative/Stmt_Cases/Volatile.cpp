// RUN: %amp_device -D__KALMAR_ACCELERATOR__ %s -emit-llvm -c -S -O2 -o %t.ll 2>&1 | %FileCheck --strict-whitespace %s

//////////////////////////////////////////////////////////////////////////////////
// Do not delete or add any line; it is referred to by absolute line number in the
// FileCheck lines below
//////////////////////////////////////////////////////////////////////////////////
#include <hc.hpp>
using namespace hc;

// volatile is not allowed in amp context
class s_volatile
{
public:
    int i;
    double d;
    unsigned long ul;
    float f;
};

void f_volatile() restrict(auto)
{
  int a = 0;
  double d = 0;
  volatile int &pi1 = (volatile int&)a;
  volatile double &pd1 = (volatile double&)d;
}

void AMP_AND_CPU_Func() [[cpu, hc]] {
  f_volatile();
}
// CHECK: Volatile.cpp:[[@LINE-2]]:3: error: call from AMP-restricted function to CPU-restricted function
// CHECK-NEXT: f_volatile();
// CHECK-NEXT: ^

int main(void)
{
  return 0;
}

