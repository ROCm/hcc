// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int fooAMP() restrict(amp)
{
  auto a_lambda = []() restrict(cpu) {}; // OK
  return 1;
}


int main(void)
{
   parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    auto a_lambda = []() restrict(cpu) {};// OK
  });
  return 0;
}

