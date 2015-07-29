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

// test find
template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  // randomly pick one value to be found
  int indexRandom = dis(gen);
  _Tp value = table[indexRandom];

  // launch kernel with parallel STL find
  using namespace std::experimental::parallel;
  auto result = find(par, std::begin(table), std::end(table), value);

  // verify data
  bool ret = (std::distance(std::begin(table), result) == indexRandom);

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

// test find_if
template<typename _Tp, size_t SIZE>
bool test2() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  // randomly pick one value to be found
  int indexRandom = dis(gen);
  _Tp value = table[indexRandom];

  // launch kernel with parallel STL find_if
  using namespace std::experimental::parallel;
  // use a custom predicate
  auto result = find_if(par, std::begin(table), std::end(table), 
                        [&](const _Tp& a) { return (a == value); });

  // verify data
  bool ret = (std::distance(std::begin(table), result) == indexRandom);

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << "," << table2[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

// test find_if_not
template<typename _Tp, size_t SIZE>
bool test3() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  // randomly pick one value to be found
  int indexRandom = dis(gen);
  _Tp value = table[indexRandom];

  // launch kernel with parallel STL find_if_not
  using namespace std::experimental::parallel;
  // use a custom predicate
  auto result = find_if_not(par, std::begin(table), std::end(table), 
                            [&](const _Tp& a) { return (a != value); });

  // verify data
  bool ret = (std::distance(std::begin(table), result) == indexRandom);

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << table[i * COL + j] << "," << table2[i * COL + j] << " ";
    }
    std::cout << "\n";
  } 
#endif

  return ret;
}

int main() {
  bool ret = true;

  // find
  ret &= test<int, TEST_SIZE>();
  ret &= test<unsigned, TEST_SIZE>();
  ret &= test<float, TEST_SIZE>();
  ret &= test<double, TEST_SIZE>();

  // find_if
  ret &= test2<int, TEST_SIZE>();
  ret &= test2<unsigned, TEST_SIZE>();
  ret &= test2<float, TEST_SIZE>();
  ret &= test2<double, TEST_SIZE>();

  // find_if_not
  ret &= test3<int, TEST_SIZE>();
  ret &= test3<unsigned, TEST_SIZE>();
  ret &= test3<float, TEST_SIZE>();
  ret &= test3<double, TEST_SIZE>();

  return !(ret == true);
}

