// RUN: %cxxamp %s -o %t.out && %t.out
// XFAIL: *

#include <hc.hpp>
using namespace hc;

int CPU_Func()
{
  auto a_lambda = []() [[hc]]
  {

  };
  return 0;
}

int CPU_Func_1() [[cpu]]
{
  auto a_lambda = []() [[hc]]
  {

  };
  return 1;
}

inline
int CPU_Func_X()
{
  parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
  {
    // OK
  });
  return 0;
}

inline
int CPU_Func_Y() [[cpu]]
{
  parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
  {
    // OK
  });
  return 1;
}


int main(void)
{
  // This test outlines a subtle issue with how we obtain mangled kernel names
  // which is tracked in SWDEV-137849. CPU_Func_X and CPU_Func_Y are made
  // inline to work around this and ensure matched mangling.
  CPU_Func();
  CPU_Func_1();
  CPU_Func_X();
  CPU_Func_Y();

  auto a_lambda = [] () [[cpu]] {
    parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
    {
      // OK
    });
  };

  auto a_lambda_1 = [] () [[cpu]] {
    auto a_lambda_AMP = [] () [[hc]] {}; //OK
  };
  return 0;
}

