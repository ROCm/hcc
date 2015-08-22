// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
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
  _Tp output[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input1), std::end(input1), 0);
  std::iota(std::begin(input2), std::end(input2), 0);

  // test kernel
  auto f = [](const _Tp& v1, const _Tp& v2)
  {
    return v1 + v2;
  };

  // launch kernel with parallel STL transform
  using namespace std::experimental::parallel;
  auto iter = transform(par, std::begin(input1), std::end(input1),
                 std::begin(input2),
                 std::begin(output), f);

  // verify data
  bool ret = true;
  if (iter != std::begin(output) + SIZE) {
    ret = false;
  }
  for (int i = 0; i < SIZE; ++i) {
    if (output[i] != i * 2)  {
      ret = false;
      break;
    }
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output[i * COL + j];
    }
    std::cout << "\n";
  } 
#endif

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

