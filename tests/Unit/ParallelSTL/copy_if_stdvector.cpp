
// RUN: %hc %s -o %t.out && %t.out

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
  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([pred](stdVector &input, stdVector &output1,
                                                                      stdVector &output2) {
    std::copy_if(std::begin(input), std::end(input), std::begin(output1), pred);
    std::experimental::parallel::
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

