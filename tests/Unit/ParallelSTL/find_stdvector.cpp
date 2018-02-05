
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
  // std::vector
  typedef std::vector<T> stdVector;
  // find
  ret &= run_and_compare<T, SIZE, stdVector>([&eq]
                                             (stdVector &input, stdVector &output1,
                                                                stdVector &output2) {
    auto expected = std::find(std::begin(input), std::end(input), T{});
    auto result   = std::experimental::parallel::
                    find(par, std::begin(input), std::end(input), T{});

    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // find_if
  ret &= run_and_compare<T, SIZE, stdVector>([&eq]
                                             (stdVector &input, stdVector &output1,
                                                                stdVector &output2) {
    auto pred = [=](const T& a) { return (a == input[12]); };

    auto expected = std::find_if(std::begin(input), std::end(input), pred);
    auto result   = std::experimental::parallel::
                    find_if(par, std::begin(input), std::end(input), pred);

    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // find_if_not
  ret &= run_and_compare<T, SIZE, stdVector>([&eq]
                                             (stdVector &input, stdVector &output1,
                                                                stdVector &output2) {
    auto pred = [=](const T& a) { return (a != input[12]); };

    auto expected = std::find_if_not(std::begin(input), std::end(input), pred);
    auto result   = std::experimental::parallel::
                    find_if_not(par, std::begin(input), std::end(input), pred);

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

