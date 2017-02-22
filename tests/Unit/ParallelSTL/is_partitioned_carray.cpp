
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

// negative test
// is_paritioned is expected to return false
template<typename _Tp, size_t SIZE>
bool test_negative() {

  _Tp table[SIZE] { 0 };

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  int indexRandom = dis(gen);

  // initialize test data
  std::fill(std::begin(table), std::begin(table) + indexRandom, 1);
  std::fill(std::begin(table) + indexRandom, std::end(table), 0);

  // launch kernel with parallel STL is_partitioned
  using namespace std::experimental::parallel;
  bool result = is_partitioned(par, std::begin(table), std::end(table), [](const _Tp& a) { return int(a) % 2 == 0; });

  // verify data
  bool ret = (result == false);

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

// positive test
// is_partitioned is expected to return true
template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  int indexRandom = dis(gen);

  // initialize test data
  std::fill(std::begin(table), std::begin(table) + indexRandom, 0);
  std::fill(std::begin(table) + indexRandom, std::end(table), 1);

  // launch kernel with parallel STL is_partitioned
  using namespace std::experimental::parallel;
  bool result = is_partitioned(par, std::begin(table), std::end(table), [](const _Tp& a) { return int(a) % 2 == 0; });

  // verify data
  bool ret = (result == true);

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

  // positive tests
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  // negative tests
  ret &= test_negative<int, TEST_SIZE>();
  ret &= test_negative<unsigned, TEST_SIZE>();
  ret &= test_negative<float, TEST_SIZE>();
  ret &= test_negative<double, TEST_SIZE>();

  return !(ret == true);
}

