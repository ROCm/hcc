// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

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
  using namespace std::experimental::parallel;

  auto pred = [](const T& a) { return int(a) % 2 == 0; };

  bool ret = true;
  bool eq = true;
  ret &= run<T, SIZE>([&eq, pred]
                      (T (&input1)[SIZE], T (&input2)[SIZE]) {
    std::partition(std::begin(input1), std::end(input1), pred);
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

