// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
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

int CPU_Func_X()
{
  parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
  {
    // OK
  });
  return 0;
}

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

