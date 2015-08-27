// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  using namespace std::experimental::parallel;

  bool ret = true;
  bool eq = true;
  ret &= run<T, SIZE>([&eq]
                      (T (&input1)[SIZE],
                       T (&input2)[SIZE]) {
    auto expected = std::accumulate(std::begin(input1), std::end(input1), T{});
    auto result   = reduce(par, std::begin(input2), std::end(input2), T{});

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

  ret &= run<T, SIZE>([&eq]
                      (T (&input1)[SIZE],
                       T (&input2)[SIZE]) {
    auto expected = std::accumulate(std::begin(input1), std::end(input1), 10);
    auto result   = reduce(par, std::begin(input2), std::end(input2), 10);

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

