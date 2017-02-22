
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

#define _DEBUG (0)

// test unique_copy (non-predicated version)
template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp table2[SIZE] { 0 };
  _Tp n { 0 };

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::bernoulli_distribution dis(0.5);

  // initialize test data
  for (int i = 1; i < SIZE; ++i) {
    if (dis(gen)) {
      table[i] = table[i - 1];
    } else {
      table[i] = n++;
    }
  }

  // launch kernel with parallel STL unique_copy
  using namespace std::experimental::parallel;
  auto result = unique_copy(par, std::begin(table), std::end(table), std::begin(table2));

  // verify data
  bool ret = true;
  ret = (std::distance(std::begin(table2), result) == n);
  for (int i = 0; i < n; ++i) {
    if (table2[i] != i) {
      ret = false;
      break;
    }
  }
  for (int i = n; i < SIZE; ++i) {
    if (table2[i] != 0) {
      ret = false;
      break;
    }
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table2[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

// test unique_copy (predicated version)
template<typename _Tp, size_t SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp table2[SIZE] { 0 };
  _Tp n { 0 };

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::bernoulli_distribution dis(0.5);

  // initialize test data
  for (int i = 1; i < SIZE; ++i) {
    if (dis(gen)) {
      table[i] = table[i - 1];
    } else {
      table[i] = n++;
    }
  }

  // launch kernel with parallel STL unique_copy
  using namespace std::experimental::parallel;

  // use custom predicate
  auto pred = [](const _Tp& a, const _Tp& b) { return a == b; };
  auto result = unique_copy(par, std::begin(table), std::end(table), std::begin(table2), pred);

  // verify data
  bool ret = true;
  ret = (std::distance(std::begin(table2), result) == n);
  for (int i = 0; i < n; ++i) {
    if (table2[i] != i) {
      ret = false;
      break;
    }
  }
  for (int i = n; i < SIZE; ++i) {
    if (table2[i] != 0) {
      ret = false;
      break;
    }
  }

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

  // unique_copy (non-predicated version)
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  // unique_copy (predicated version)
  ret &= test2<int, TEST_SIZE>();
  ret &= test2<unsigned, TEST_SIZE>();
  ret &= test2<float, TEST_SIZE>();
  ret &= test2<double, TEST_SIZE>();

  return !(ret == true);
}

