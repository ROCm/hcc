// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>

int f1() [[cpu, hc]] {return 1;} 
int f2() restrict(auto) {
  static int i;
  return f1();
}

int AMP_AND_CPU_Func() [[cpu, hc]] 
{
  f2();  // OK. 'auto' is inferred to (cpu,amp)
  return 1;
}

int main(void)
{
  return 0;  // expected: success
}

