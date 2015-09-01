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


  auto pred = [](const T& a) { return int(a) % 2 == 0; };
  using std::experimental::parallel::par;

  bool ret = true;
  ret &= run_and_compare<T, SIZE>([pred]
                                  (T (&input1)[SIZE],
                                   T (&input2)[SIZE]) {
    std::stable_partition(std::begin(input1), std::end(input1), pred);
    std::experimental::parallel::
    stable_partition(par, std::begin(input2), std::end(input2), pred);

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

