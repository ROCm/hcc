
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

// test find_first_of (non-predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp pattern1[PATTERN_SIZE] { 0 }; // this pattern contains elements in table so is expected to be found
  _Tp pattern2[PATTERN_SIZE] { 0 }; // this pattern contains no elements in table so is not expected to be found
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);
  std::uniform_int_distribution<int> dis2(SIZE + 1, SIZE * 2);

  // randomly pick values in table and set to pattern1
  int smallestIndex = SIZE + 1;
  int indexRandom = 0;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    indexRandom = dis(gen);
    pattern1[i] = table[indexRandom];
    if (smallestIndex > indexRandom)
      smallestIndex = indexRandom;
  }

  // randomly pick values to pattern2
  // none of the values fall in the range of values in table
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern2[i] = _Tp(dis2(gen));
  }

  // launch kernel with parallel STL find_first_of
  using namespace std::experimental::parallel;

  // for pattern1 we expect the element from smallestIndex is found
  auto result1 = find_first_of(par, std::begin(table), std::end(table), std::begin(pattern1), std::end(pattern1));
  // for pattern2 we expect none of the element is found
  auto result2 = find_first_of(par, std::begin(table), std::end(table), std::begin(pattern2), std::end(pattern2));

  // verify data
  bool ret =
    (std::distance(std::begin(table), result1) == smallestIndex) &&
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

// test find_first_of (predicated version)
template<typename _Tp, size_t SIZE, size_t PATTERN_SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp pattern1[PATTERN_SIZE] { 0 }; // this pattern contains elements in table so is expected to be found
  _Tp pattern2[PATTERN_SIZE] { 0 }; // this pattern contains no elements in table so is not expected to be found
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);
  std::uniform_int_distribution<int> dis2(SIZE + 1, SIZE * 2);

  // randomly pick values in table and set to pattern1
  int smallestIndex = SIZE + 1;
  int indexRandom = 0;
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    indexRandom = dis(gen);
    pattern1[i] = table[indexRandom];
    if (smallestIndex > indexRandom)
      smallestIndex = indexRandom;
  }

  // randomly pick values to pattern2
  // none of the values fall in the range of values in table
  for (int i = 0; i < PATTERN_SIZE; ++i) {
    pattern2[i] = _Tp(dis2(gen));
  }

  // launch kernel with parallel STL find_first_of
  using namespace std::experimental::parallel;

  // use custom predicate
  auto pred = [](const _Tp& a, const _Tp& b) { return a == b; };

  // for pattern1 we expect the element from smallestIndex is found
  auto result1 = find_first_of(par, std::begin(table), std::end(table), std::begin(pattern1), std::end(pattern1), pred);
  // for pattern2 we expect none of the element is found
  auto result2 = find_first_of(par, std::begin(table), std::end(table), std::begin(pattern2), std::end(pattern2), pred);

  // verify data
  bool ret =
    (std::distance(std::begin(table), result1) == smallestIndex) &&
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

  // non-predicated find_first_of
  ret &= test<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  // predicated find_first_of
  ret &= test2<int, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<unsigned, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<float, TEST_SIZE, TEST_PATTERN_SIZE>();
  ret &= test2<double, TEST_SIZE, TEST_PATTERN_SIZE>();

  return !(ret == true);
}

