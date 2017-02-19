
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
  auto f = [&](T& v) [[hc,cpu]]
  {
    v *= 8;
    v += 3;
  };

  using std::experimental::parallel::par;

  bool ret = true;
  ret &= run_and_compare<T, SIZE>([f](T (&input1)[SIZE],
                                      T (&input2)[SIZE]) {
    std::for_each(std::begin(input1), std::end(input1), f);
    std::experimental::parallel::
    for_each(par, std::begin(input2), std::end(input2), f);
  });

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<int, ROW * COL>();
  ret &= test<float, ROW * COL>();

  return !(ret == true);
}

