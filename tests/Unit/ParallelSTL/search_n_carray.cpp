
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

// test search_n (non-predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE * 2);
  std::uniform_int_distribution<int> dis2(1, PATTERN_SIZE);

  // randomly pick the number of elements to be made identical
  int patternSize = dis2(gen);

  // randomly pick one index
  int indexRandom = dis(gen);
  _Tp value;
  if ((indexRandom >= 0) && (indexRandom < SIZE - patternSize)) {
    // in case the index falls in the domain of table
    // make elements be identical
    value = table[indexRandom];
    for (int i = 0; i < patternSize; ++i) {
      table[indexRandom + i] = table[indexRandom];
    }
  } else {
    // in case the index falls out of the domain the table
    // do nothing
    value = _Tp(indexRandom);
  }

  // launch kernel with parallel STL search_n
  using namespace std::experimental::parallel;
  auto result = search_n(par, std::begin(table), std::end(table), patternSize, value);

  // verify data
  bool ret = true;
  if ((indexRandom >= 0) && (indexRandom < SIZE - patternSize)) {
    ret = (std::distance(std::begin(table), result) == indexRandom);
  } else {
    ret = (result == std::end(table));
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

// test search_n (predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE * 2);
  std::uniform_int_distribution<int> dis2(1, PATTERN_SIZE);

  // randomly pick the number of elements to be made identical
  int patternSize = dis2(gen);

  // randomly pick one index
  int indexRandom = dis(gen);
  _Tp value;
  if ((indexRandom >= 0) && (indexRandom < SIZE - patternSize)) {
    // in case the index falls in the domain of table
    // make elements be identical
    value = table[indexRandom];
    for (int i = 0; i < patternSize; ++i) {
      table[indexRandom + i] = table[indexRandom];
    }
  } else {
    // in case the index falls out of the domain the table
    // do nothing
    value = _Tp(indexRandom);
  }

  // launch kernel with parallel STL search_n
  using namespace std::experimental::parallel;

  // use custom predicate
  auto pred = [](const _Tp& a, const _Tp& b) { return a == b; };
  auto result = search_n(par, std::begin(table), std::end(table), patternSize, value, pred);

  // verify data
  bool ret = true;
  if ((indexRandom >= 0) && (indexRandom < SIZE - patternSize)) {
    ret = (std::distance(std::begin(table), result) == indexRandom);
  } else {
    ret = (result == std::end(table));
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

  // search_n (non-predicated version)
  ret &= test<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  // search_n (predicated version)
  ret &= test2<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  return !(ret == true);
}

