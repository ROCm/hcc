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
  // find
  ret &= run<T, SIZE>([&eq]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {
    auto expected = std::find(std::begin(input), std::end(input), T{});
    auto result   = find(par, std::begin(input), std::end(input), T{});

    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // find_if
  ret &= run<T, SIZE>([&eq]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {
    auto pred = [=](const T& a) { return (a == input[12]); };

    auto expected = std::find_if(std::begin(input), std::end(input), pred);
    auto result   = find_if(par, std::begin(input), std::end(input), pred);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // find_if_not
  ret &= run<T, SIZE>([&eq]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {

    auto pred = [=](const T& a) { return (a != input[12]); };

    auto expected = std::find_if_not(std::begin(input), std::end(input), pred);
    auto result   = find_if_not(par, std::begin(input), std::end(input), pred);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

  return ret;
}

int main() {
  bool ret = true;

  // find
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  return !(ret == true);
}

