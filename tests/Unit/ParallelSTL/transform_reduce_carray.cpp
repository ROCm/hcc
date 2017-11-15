
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

  auto op = [](const T &x) [[hc,cpu]] { return x+5566; };
  auto binary_op = std::plus<T>();
  auto init = T{};

  using std::experimental::parallel::par;

  bool ret = true;
  bool eq = true;
  ret &= run_and_compare<T, SIZE>([op, init, binary_op, &eq]
                                  (T (&input)[SIZE], T (&output1)[SIZE],
                                                     T (&output2)[SIZE]) {
    std::transform(std::begin(input), std::end(input), std::begin(output1), op);
    auto expected = std::accumulate(std::begin(output1), std::end(output1), init, binary_op);

    auto result = std::experimental::parallel::
                  transform_reduce(par, std::begin(input), std::end(input), op, init, binary_op);
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

