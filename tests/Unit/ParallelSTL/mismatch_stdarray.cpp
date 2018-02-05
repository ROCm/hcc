
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
  bool eq = true;

  // std::array
  typedef std::array<T, SIZE> stdArray;
  // test mismatch (non-predicated version)
  ret &= run_and_compare<T, SIZE, stdArray>([&eq]
                                            (stdArray &input1, stdArray &input2, stdArray &output1,
                                                                                 stdArray &output2) {
    auto expected = std::mismatch(std::begin(input1), std::end(input2), std::begin(input2));
    auto result   = std::experimental::parallel::
                    mismatch(par, std::begin(input1), std::end(input2), std::begin(input2));

    eq = expected == result;
  }, false);
  ret &= eq;

  auto pred = [](const T &a, const T &b) { return ((a + 1) == b); };

  ret &= run_and_compare<T, SIZE, stdArray>([&eq, pred]
                                            (stdArray &input1, stdArray &input2, stdArray &output1,
                                                                                 stdArray &output2) {
    auto expected = std::mismatch(std::begin(input1), std::end(input2), std::begin(input2), pred);
    auto result   = std::experimental::parallel::
                    mismatch(par, std::begin(input1), std::end(input2), std::begin(input2), pred);

    eq = expected == result;
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

