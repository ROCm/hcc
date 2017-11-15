
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  using std::experimental::parallel::par;

  bool ret = true;
  // C array
  typedef T cArray[SIZE];
  ret &= run_and_compare<T, SIZE>([](cArray &input1,
                                     cArray &input2) {
    std::fill(std::begin(input1), std::end(input1), SIZE + 1);
    std::experimental::parallel::
    fill(par, std::begin(input2), std::end(input2), SIZE + 1);
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

