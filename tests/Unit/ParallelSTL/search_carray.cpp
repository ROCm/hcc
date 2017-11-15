
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define TEST_PATTERN_SIZE (4)

#define _DEBUG (0)

// test search (non-predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test() {

  static_assert(SIZE % PATTERN_SIZE == 0, "PATTERN_SIZE doesn't divide SIZE");

  _Tp table[SIZE] { 0 };
  _Tp pattern1[PATTERN_SIZE] { 0 }; // this pattern is expected to be found
  _Tp pattern2[PATTERN_SIZE] { 0 }; // this pattern is expected not to be found
  _Tp n { 0 };

  // initialize test data
  for (int i = 0; i < SIZE; ++i) {
    table[i] = n++;
    if (n == PATTERN_SIZE) n = 0;
  }
  n = 0;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern1[i] = n++;
  }
  n = PATTERN_SIZE;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern2[i] = n++;
  }

  // launch kernel with parallel STL search
  using namespace std::experimental::parallel;

  // for pattern1 we expect the pattern is found
  auto result1 = search(par, std::begin(table), std::end(table), std::begin(pattern1), std::end(pattern1));
  // for pattern2 we expect the pattern is not found
  auto result2 = search(par, std::begin(table), std::end(table), std::begin(pattern2), std::end(pattern2));

  // verify data
  bool ret =
    (std::distance(std::begin(table), result1) == 0) &&
    (result2 == std::end(table));

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

// test search (predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test2() {

  static_assert(SIZE % PATTERN_SIZE == 0, "PATTERN_SIZE doesn't divide SIZE");

  _Tp table[SIZE] { 0 };
  _Tp pattern1[PATTERN_SIZE] { 0 }; // this pattern is expected to be found
  _Tp pattern2[PATTERN_SIZE] { 0 }; // this pattern is expected not to be found
  _Tp n { 0 };

  // initialize test data
  for (int i = 0; i < SIZE; ++i) {
    table[i] = n++;
    if (n == PATTERN_SIZE) n = 0;
  }
  n = 0;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern1[i] = n++;
  }
  n = PATTERN_SIZE;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern2[i] = n++;
  }

  // launch kernel with parallel STL search
  using namespace std::experimental::parallel;

  // use custom predicate
  auto pred = [](const _Tp& a, const _Tp& b) { return a == b; };

  // for pattern1 we expect the pattern is found
  auto result1 = search(par, std::begin(table), std::end(table), std::begin(pattern1), std::end(pattern1), pred);
  // for pattern2 we expect the pattern is not found
  auto result2 = search(par, std::begin(table), std::end(table), std::begin(pattern2), std::end(pattern2), pred);

  // verify data
  bool ret =
    (std::distance(std::begin(table), result1) == 0) &&
    (result2 == std::end(table));

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

int main() {
  bool ret = true;

  // non-predicated search
  ret &= test<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  // predicated search
  ret &= test2<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  return !(ret == true);
}

