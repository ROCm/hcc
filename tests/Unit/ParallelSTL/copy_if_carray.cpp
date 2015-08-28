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

  using namespace std::experimental::parallel;

  bool ret = true;
  ret &= run<T, SIZE>([pred](T (&input)[SIZE], T (&output1)[SIZE],
                                                T (&output2)[SIZE]) {
    std::copy_if(std::begin(input), std::end(input), std::begin(output1), pred);
    copy_if(par, std::begin(input), std::end(input), std::begin(output2), pred);
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

