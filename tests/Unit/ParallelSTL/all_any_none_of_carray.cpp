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
#include <numeric>
#include <algorithm>
#include <iterator>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {
  bool ret = true;

  _Tp table[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(table), std::end(table), 0);

  auto pred1 = [](const _Tp& v) { return static_cast<int>(v) % 3 == 0; };
  auto expected = std::any_of(std::begin(table), std::end(table), pred1);

  // launch kernel with parallel STL any_of
  auto result = std::experimental::parallel::any_of(std::begin(table), std::end(table), pred1);
#if _DEBUG
  std::cout << std::boolalpha << expected << " " << result << "\n";
#endif
  ret &= (expected == result);


  auto pred2 = [](const _Tp& v) { return static_cast<int>(v) == SIZE + 1; };
  expected = std::none_of(std::begin(table), std::end(table), pred2);

  // launch kernel with parallel STL none_of
  result = std::experimental::parallel::none_of(std::begin(table), std::end(table), pred2);
#if _DEBUG
  std::cout << std::boolalpha << expected << " " << result << "\n";
#endif
  ret &= (expected == result);


  auto pred3 = [](const _Tp& v) { return v >= 0; };
  expected = std::all_of(std::begin(table), std::end(table), pred3);

  // launch kernel with parallel STL all_of
  result = std::experimental::parallel::all_of(std::begin(table), std::end(table), pred3);
#if _DEBUG
  std::cout << std::boolalpha << expected << " " << result << "\n";
#endif
  ret &= (expected == result);

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

