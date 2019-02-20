// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;

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
int fooCPU() [[cpu]]  // OK
{
  foo2();  // OK
  return 0;
}


int main(void)
{
  parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
    {
        fooAMP();
    });
}

