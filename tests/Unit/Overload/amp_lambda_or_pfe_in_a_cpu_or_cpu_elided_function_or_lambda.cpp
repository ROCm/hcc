// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

int CPU_Func()
{
  auto a_lambda = []() restrict(amp)
  {

  };
  return 0;
}

int CPU_Func_1() restrict(cpu)
{
  auto a_lambda = []() restrict(amp)
  {

  };
  return 1;
}

inline
int CPU_Func_X()
{
  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    // OK
  });
  return 0;
}

inline
int CPU_Func_Y() restrict(cpu)
{
  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
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

  auto a_lambda = [] () restrict(cpu) {
    parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
      // OK
    });
  };

  auto a_lambda_1 = [] () restrict(cpu) {
    auto a_lambda_AMP = [] () restrict(amp) {}; //OK
  };
  return 0;
}

