// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  using namespace std::experimental::parallel;

  bool ret = true;
  bool eq = true;

  // test mismatch (non-predicated version)
  ret &= run<T, SIZE>([&eq]
                      (T (&input1)[SIZE], T (&input2)[SIZE], T (&output1)[SIZE],
                                                             T (&output2)[SIZE]) {
    auto expected = std::mismatch(std::begin(input1), std::end(input2), std::begin(input2));
    auto result   = mismatch(par, std::begin(input1), std::end(input2), std::begin(input2));

    eq = expected == result;
  }, false);
  ret &= eq;

  auto pred = [](const T &a, const T &b) { return ((a + 1) == b); };

  ret &= run<T, SIZE>([&eq, pred]
                      (T (&input1)[SIZE], T (&input2)[SIZE], T (&output1)[SIZE],
                                                             T (&output2)[SIZE]) {
    auto expected = std::mismatch(std::begin(input1), std::end(input2), std::begin(input2), pred);
    auto result   = mismatch(par, std::begin(input1), std::end(input2), std::begin(input2), pred);

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

