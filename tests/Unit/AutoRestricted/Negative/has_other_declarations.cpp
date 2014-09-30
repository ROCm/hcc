//XFAIL:*
// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2() restrict(cpu);        // expected-note{{previous declaration is here}}
int f2() restrict(cpu,auto) {  // expected-error{{'f2':  expected no other declaration since it is auto restricted}}
  return f1();
}
int main(void)
{
  f2();
  return 0;  // should not compile
}

