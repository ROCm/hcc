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

  auto f = [] { return SIZE + 1; };
  using std::experimental::parallel::par;

  bool ret = true;
  ret &= run_and_compare<T, SIZE>([f](T (&input1)[SIZE],
                                      T (&input2)[SIZE]) {
    std::generate(std::begin(input1), std::end(input1), f);
    std::experimental::parallel::
    generate(par, std::begin(input2), std::end(input2), f);
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

