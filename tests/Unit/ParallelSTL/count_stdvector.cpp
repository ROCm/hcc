
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <iterator>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  using std::experimental::parallel::par;

  bool ret = true;
  bool eq = true;
  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([&eq](stdVector &input, stdVector &output1,
                                                                     stdVector &output2) {
    auto expected = std::count(std::begin(input), std::end(input), 42);
    auto result   = std::experimental::parallel::
                    count(par, std::begin(input), std::end(input), 42);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;



  auto pred = [](const T& v) [[hc,cpu]] { return static_cast<int>(v) % 3 == 0; };

  ret &= run_and_compare<T, SIZE, stdVector>([&eq, pred](stdVector &input, stdVector &output1,
                                                                           stdVector &output2) {
    auto expected = std::count_if(std::begin(input), std::end(input), pred);
    auto result   = std::experimental::parallel::
                    count_if(par, std::begin(input), std::end(input), pred);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

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

