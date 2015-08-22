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

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  auto first = std::begin(table);
  auto last = std::end(table);
  auto mid = std::next(first, std::distance(first, last) / 2);
  std::generate(first, mid, [] { return 1; });
  std::generate(mid, last, [] { return 2; });

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  replace(par, std::begin(table), std::end(table), 2, 3);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE / 2; ++i) {
    if (table[i] != 1) {
      ret = false;
      break;
    }
  }
  for (int i = SIZE / 2; i < SIZE; ++i) {
    if (table[i] != 3) {
      ret = false;
      break;
    }
  }

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

