//XFAIL:*
// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

int f1() restrict(cpu,amp) {return 1;} 
int f2xx() restrict(cpu,auto);  // expected-error{{'auto' restriction specifier is only allowed on function definition}}
int f2xx() restrict(cpu)
{
  return f1();
}
int main(void)
{
  f2xx();
  return 0;  // should not compile
}

