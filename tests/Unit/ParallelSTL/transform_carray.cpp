// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename _Tp, size_t SIZE>
bool test() {
  // test kernel
  auto f = [&](_Tp& v)
  {
    return v * 2;
  };

  using namespace std::experimental::parallel;

  bool ret = true;
  ret &= run<_Tp, SIZE>([f](_Tp *input1, _Tp *output1,
                            _Tp *input2, _Tp *output2) {
    std::transform(input1, input1+SIZE, output1, f);
    transform(par, input2, input2+SIZE, output2, f);
  });

  // test kernel 2
  auto g = [&](_Tp& v)
  {
    return v + 5566;
  };

  ret &= run<_Tp, SIZE>([g](_Tp *input1, _Tp *output1,
                            _Tp *input2, _Tp *output2) {
    std::transform(input1, input1+SIZE, output1, g);
    transform(par, input2, input2+SIZE, output2, g);
  });

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  return !(ret == true);
}

