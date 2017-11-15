
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

  // C array
  typedef T cArray[SIZE];
  ret &= run_and_compare<T, SIZE>([&eq]
                                  (cArray &input1, cArray &input2) {
    std::stable_sort(std::begin(input1), std::end(input1));
    std::experimental::parallel::
    stable_sort(par, std::begin(input2), std::end(input2));

    eq = std::equal(std::begin(input1), std::end(input1), std::begin(input2));
  }, false);
  ret &= eq;

  ret &= run_and_compare<T, SIZE>([&eq]
                                  (cArray &input1, cArray &input2) {
    std::stable_sort(std::begin(input1), std::end(input1));
    std::experimental::parallel::
    stable_sort(par, std::begin(input2), std::end(input2));

    eq = std::equal(std::begin(input1), std::end(input1), std::begin(input2));
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

