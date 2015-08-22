// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <iterator>

#define ROW (8)
#define COL (12)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {
  bool ret = true;

  _Tp table[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(table), std::end(table), 0);

  // launch kernel with parallel STL count
  auto expected = std::count(std::begin(table), std::end(table), 377);

  ret &= (expected == std::experimental::parallel::count(std::begin(table), std::end(table), 377));

  auto pred = [](const _Tp& v) { return static_cast<int>(v) % 3 == 0; };

  expected = std::count_if(std::begin(table), std::end(table), pred);
  ret &= (expected == std::experimental::parallel::count_if(std::begin(table), std::end(table), pred));

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j];
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

