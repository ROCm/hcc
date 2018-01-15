// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int fooCPU() restrict(cpu)
{
  return 1;
}

int foo()
{
  return 2;
}

int main(void)
{
  fooCPU();
  foo();
  auto a_lambda = [] () restrict(cpu) {};
  auto another_lambda = [] () {};

  return 0;
}

