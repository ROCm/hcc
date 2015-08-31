// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


// test adjacent_find (non-predicated version)
template<typename T, size_t SIZE>
bool test(void) {

  using namespace std::experimental::parallel;

  bool ret = true;
  bool eq = true;
  ret &= run<T, SIZE>([&eq]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {
    auto expected = std::adjacent_find(std::begin(input), std::end(input));
    auto result   = adjacent_find(par, std::begin(input), std::end(input));

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

  ret &= run<T, SIZE>([&eq]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {
    // make them equal
    input[2] = input[3];
    auto expected = std::adjacent_find(std::begin(input), std::end(input));
    auto result   = adjacent_find(par, std::begin(input), std::end(input));

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

  // use custom predicate
  auto pred = [](const T& a, const T& b) { return a > b; };

  ret &= run<T, SIZE>([&eq, pred]
                      (T (&input)[SIZE], T (&output1)[SIZE],
                                         T (&output2)[SIZE]) {
    // test adjacent_find (predicated version)
    auto expected = std::adjacent_find(std::begin(input), std::end(input), pred);
    auto result   = adjacent_find(par, std::begin(input), std::end(input), pred);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;

  return ret;
}

int main() {
  bool ret = true;

  // adjacent_find (non-predicated version)
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();


  return !(ret == true);
}

