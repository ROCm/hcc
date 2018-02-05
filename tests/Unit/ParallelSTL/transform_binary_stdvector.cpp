
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"

template<typename T, size_t SIZE>
bool test(void) {
  // test kernel
  auto f = [](const T& v1, const T& v2) [[hc,cpu]] { return v1 + v2; };
  auto g = [](const T& v1, const T& v2) [[hc,cpu]] { return v1 - v2; };

  using std::experimental::parallel::par;

  bool ret = true;

  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([f](stdVector &input1, stdVector &input2, stdVector &output1,
                                                                                       stdVector &output2) {
    std::transform(std::begin(input1), std::end(input1), std::begin(input2), std::begin(output1), f);
    std::experimental::parallel::
    transform(par, std::begin(input1), std::end(input1), std::begin(input2), std::begin(output2), f);
  });

  ret &= run_and_compare<T, SIZE, stdVector>([g](stdVector &input1, stdVector &input2, stdVector &output1,
                                                                                       stdVector &output2) {
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

