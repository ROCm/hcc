// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;


inline
int fooAMP() [[hc]]
{
  auto a_lambda = []() [[cpu]] {}; // OK
  return 1;
}


int main(void)
{
  // This test outlines a subtle issue with how we obtain mangled kernel names
  // which is tracked in SWDEV-137849. fooAMP is made inline to work around this
  // and ensure matched mangling.
   parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
  {
    auto a_lambda = []() [[cpu]] {};// OK
  });
  return 0;
}

