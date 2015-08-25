// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {

  _Tp input1[SIZE] { 0 };
  _Tp input2[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 0);
  std::iota(std::begin(input2), std::end(input2), 0);

  // test kernel
  auto f = [](const _Tp& v1, const _Tp& v2)
  {
    return v1 * v2+1;
  };

  auto expect = std::inner_product(std::begin(input1), std::end(input1),
                 std::begin(input2), _Tp{}, std::plus<_Tp>(), f);

  // launch kernel with parallel STL inner_product
  using namespace std::experimental::parallel;


  auto ans =  inner_product(par, std::begin(input1), std::end(input1),
                 std::begin(input2), _Tp{}, std::plus<_Tp>(), f);

  // verify data
  bool ret = expect == ans;
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

