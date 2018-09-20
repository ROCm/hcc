//XFAIL:*
// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>

int f1() [[cpu, hc]] {return 1;} 
int f2xx() restrict(cpu,auto);  // expected-error{{'auto' restriction specifier is only allowed on function definition}}
int f2xx() [[cpu]]
{
  return f1();
}
int main(void)
{
  f2xx();
  return 0;  // should not compile
}

