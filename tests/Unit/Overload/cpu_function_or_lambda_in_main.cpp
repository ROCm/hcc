// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;


int fooCPU() [[cpu]]
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
  auto a_lambda = [] () [[cpu]] {};
  auto another_lambda = [] () {};

  return 0;
}

