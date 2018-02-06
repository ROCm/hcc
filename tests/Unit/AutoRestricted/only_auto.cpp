// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2() restrict(auto) {
  static int i;
  return f1();
}

int AMP_AND_CPU_Func() restrict(cpu,amp) 
{
  f2();  // OK. 'auto' is inferred to (cpu,amp)
  return 1;
}

int main(void)
{
  return 0;  // expected: success
}

