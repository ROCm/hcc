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
  // test kernel
  auto f = [](const T& v1, const T& v2) { return v1 + v2; };
  auto g = [](const T& v1, const T& v2) { return v1 - v2; };

  using std::experimental::parallel::par;

  bool ret = true;
  ret &= run_and_compare<T, SIZE>([f](T (&input1)[SIZE], T (&input2)[SIZE], T (&output1)[SIZE],
                                                                            T (&output2)[SIZE]) {
    std::transform(std::begin(input1), std::end(input1), std::begin(input2), std::begin(output1), f);
    std::experimental::parallel::
    transform(par, std::begin(input1), std::end(input1), std::begin(input2), std::begin(output2), f);
  });

  ret &= run_and_compare<T, SIZE>([g](T (&input1)[SIZE], T (&input2)[SIZE], T (&output1)[SIZE],
                                                                            T (&output2)[SIZE]) {
    std::transform(std::begin(input1), std::end(input1), std::begin(input2), std::begin(output1), g);
    std::experimental::parallel::
    transform(par, std::begin(input1), std::end(input1), std::begin(input2), std::begin(output2), g);
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

