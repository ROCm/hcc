// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2() restrict(cpu,auto) {
  return f1();
}
int main(void)
{
  f2();
  return 0;  // expected: success
}

