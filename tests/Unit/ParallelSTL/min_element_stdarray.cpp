
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  using std::experimental::parallel::par;

  bool ret = true;
  bool eq = true;
  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([&eq]
                                             (stdArray &input, stdArray &output1,
                                                               stdArray &output2) {
    auto expected = std::min_element(std::begin(input), std::end(input));
    auto result   = std::experimental::parallel::
                    min_element(par, std::begin(input), std::end(input));

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

