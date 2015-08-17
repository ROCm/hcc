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

  _Tp input[SIZE] { 0 };
  _Tp output[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(input), std::end(input), [&] { return (n += 2); });

  // launch kernel with parallel STL adjacent_difference (non-predicated version)
  using namespace std::experimental::parallel;
  auto iter = adjacent_difference(par, std::begin(input), std::end(input), std::begin(output));

  // verify data
  bool ret = true;
  if (iter != std::begin(output) + SIZE) {
    ret = false;
  }
  for (int i = 0; i < SIZE; ++i) {
    if (output[i] != 2)  {
      ret = false;
      break;
    }
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << input[i * COL + j];
    }
    std::cout << "\n";
  } 
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

