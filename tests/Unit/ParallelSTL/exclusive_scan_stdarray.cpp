// XFAIL: *
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

  auto binary_op = std::plus<T>();
  auto init = T{};

  using std::experimental::parallel::par;

  bool ret = true;
  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([init, binary_op]
                                            (stdArray &input, stdArray &output1,
                                                              stdArray &output2) {
    std::partial_sum(std::begin(input), std::end(input), std::begin(output1), binary_op);
    for (int i = SIZE-2; i >= 0; i--)
      output1[i+1] = binary_op(init, output1[i]);
    output1[0] = init;
    std::experimental::parallel::
    exclusive_scan(par, std::begin(input), std::end(input), std::begin(output2), init, binary_op);
  });

  return ret;
}

int main() {
  bool ret = true;

  // XXX the test will cause soft hang right now
  // make it fail immediately for now
#if 0
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  return !(ret == true);
#else
  return !(false == true);
#endif
}

