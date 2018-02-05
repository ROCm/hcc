
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  auto op = [](const T& a, const T& b) [[hc,cpu]] { return a - b; };

  using std::experimental::parallel::par;

  bool ret = true;

  // C array
  typedef T cArray[SIZE];
  ret &= run_and_compare<T, SIZE>([op](cArray &input, cArray &output1,
                                                      cArray &output2) {
    std::adjacent_difference(std::begin(input), std::end(input), std::begin(output1), op);
    std::experimental::parallel::
    adjacent_difference(par, std::begin(input), std::end(input), std::begin(output2), op);
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

