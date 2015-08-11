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

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  replace_if(par, std::begin(table), std::end(table), [](_Tp& v) { return static_cast<int>(v) % 3 == 0; }, SIZE + 1);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if (i % 3 == 0) {
      // for items fulfilling the predicate, the value shall be changed
      if (table[i] != SIZE + 1) {
        ret = false;
        break;
      }
    } else {
      // for items not fulfilling the predicate, the value shall not be changed
      if (table[i] != i) {
        ret = false;
        break;
      }
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

