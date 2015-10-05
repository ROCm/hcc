// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

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

// test adjacent_find (non-predicated version)
template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE * 2);

  // randomly pick one index
  int indexRandom = dis(gen);
  if ((indexRandom >= 0) && (indexRandom < SIZE - 1)) {
    // in case the index falls in the domain of table
    // make the elemnt be identical with the next one
    table[indexRandom] = table[indexRandom + 1];
  } else {
    // in case the index falls out of the domain the table
    // do nothing
  }

  // launch kernel with parallel STL adjacent_find
  using namespace std::experimental::parallel;
  auto result = adjacent_find(par, std::begin(table), std::end(table));

  // verify data
  bool ret = true;
  if ((indexRandom >= 0) && (indexRandom < SIZE - 1)) {
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

// test adjacent_find (predicated version)
template<typename _Tp, size_t SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE * 2);

  // randomly pick one index
  int indexRandom = dis(gen);
  if ((indexRandom >= 0) && (indexRandom < SIZE - 1)) {
    // in case the index falls in the domain of table
    // make the elemnt be identical with the next one
    table[indexRandom] = table[indexRandom + 1];
  } else {
    // in case the index falls out of the domain the table
    // do nothing
  }

  // launch kernel with parallel STL adjacent_find
  using namespace std::experimental::parallel;

  // use custom predicate
  auto pred = [](const _Tp& a, const _Tp& b) { return a == b; };
  auto result = adjacent_find(par, std::begin(table), std::end(table), pred);

  // verify data
  bool ret = true;
  if ((indexRandom >= 0) && (indexRandom < SIZE - 1)) {
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

  // adjacent_find (non-predicated version)
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  // adjacent_find (predicated version)
  ret &= test2<int, TEST_SIZE>();
  ret &= test2<unsigned, TEST_SIZE>();
  ret &= test2<float, TEST_SIZE>();
  ret &= test2<double, TEST_SIZE>();

  return !(ret == true);
}

