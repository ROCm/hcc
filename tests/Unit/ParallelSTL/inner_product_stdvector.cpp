
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

  // test kernel
  auto f = [](const T& v1, const T& v2) [[hc,cpu]]
  {
    return 5 * v1 + 6 * v2;
  };
  auto binary_op = std::plus<T>();
  auto init = T{};

  using std::experimental::parallel::par;

  bool ret = true;
  bool eq = true;
  // std::vector
  typedef std::vector<T> stdVector;
  ret &= run_and_compare<T, SIZE, stdVector>([init, binary_op, f, &eq]
                                              (stdVector &input1, stdVector &input2, stdVector &output1,
                                                                                     stdVector &output2) {
    auto expected = std::inner_product(std::begin(input1), std::end(input1), std::begin(input2), init, binary_op, f);
    auto result   = std::experimental::parallel::
                    inner_product(par, std::begin(input1), std::end(input1), std::begin(input2), init, binary_op, f);

    eq = EQ(expected, result);

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

