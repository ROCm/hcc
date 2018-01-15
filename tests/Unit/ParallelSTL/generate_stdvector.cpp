
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {

  auto f = []() [[hc,cpu]] { return SIZE + 1; };
  using std::experimental::parallel::par;

  bool ret = true;

  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([f](stdVector &input1,
                                                 stdVector &input2) {
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

