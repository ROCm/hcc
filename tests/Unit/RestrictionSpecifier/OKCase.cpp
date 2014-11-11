// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

int foo() restrict(,)  // OK
{
  return 0;
}


int foo1() restrict(amp,)  // OK
{
  return 0;
}
int fooAMP() restrict(,amp)  // OK
{
  foo1();  // OK
  return 0;
}


int foo2() restrict(,   ,,,   ,cpu,,,,)  // OK
{
  return 0;
}
int fooCPU() restrict(cpu)  // OK
{
  foo2();  // OK
  return 0;
}


int main(void)
{
  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
        fooAMP();
    });
}

