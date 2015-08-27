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
  ret &= run<T, SIZE>([](T (&input1)[SIZE], T(&output1)[SIZE],
                         T (&input2)[SIZE], T(&output2)[SIZE]) {
    std::replace_copy(std::begin(input1), std::end(input1), std::begin(output1), 2, 3);
    replace_copy(par, std::begin(input2), std::end(input2), std::begin(output2), 2, 3);
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

