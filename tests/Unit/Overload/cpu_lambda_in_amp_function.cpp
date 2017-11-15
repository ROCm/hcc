// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


inline
int fooAMP() restrict(amp)
{
  auto a_lambda = []() restrict(cpu) {}; // OK
  return 1;
}


int main(void)
{
  // This test outlines a subtle issue with how we obtain mangled kernel names
  // which is tracked in SWDEV-137849. fooAMP is made inline to work around this
  // and ensure matched mangling.
   parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    auto a_lambda = []() restrict(cpu) {};// OK
  });
  return 0;
}

