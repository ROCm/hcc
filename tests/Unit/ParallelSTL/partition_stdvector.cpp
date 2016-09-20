
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {
  using std::experimental::parallel::par;

  auto pred = [](const T& a) { return int(a) % 2 == 0; };

  bool ret = true;
  bool eq = true;

  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([&eq, pred]
                                             (stdVector &input1, stdVector &input2) {
    std::partition(std::begin(input1), std::end(input1), pred);
    std::experimental::parallel::
    partition(par, std::begin(input2), std::end(input2), pred);
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

