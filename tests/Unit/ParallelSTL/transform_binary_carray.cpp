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
  auto f = [](const _Tp& v1, const _Tp& v2)
  {
    return v1 + v2;
  };

  using namespace std::experimental::parallel;

  bool ret = true;
  ret &= run<_Tp, SIZE>([f](_Tp *input1, _Tp *input3, _Tp *output1,
                            _Tp *input2, _Tp *input4, _Tp *output2) {
    std::transform(input1, input1+SIZE, input3, output1, f);
    transform(par, input2, input2+SIZE, input4, output2, f);
  });

  // test kernel 2
  auto g = [](const _Tp& v1, const _Tp& v2)
  {
    return v1 - v2;
  };

  ret &= run<_Tp, SIZE>([g](_Tp *input1, _Tp *input3, _Tp *output1,
                            _Tp *input2, _Tp *input4, _Tp *output2) {
    std::transform(input1, input1+SIZE, input3, output1, g);
    transform(par, input2, input2+SIZE, input4, output2, g);
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

