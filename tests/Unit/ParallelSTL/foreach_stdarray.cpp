
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>

#define ROW (8)
#define COL (16)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {

  std::array<_Tp, SIZE> table { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // test kernel
  auto f = [](_Tp& v) [[hc,cpu]]
  {
    v *= 8;
    v += 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, std::begin(table), std::end(table), f);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != i * 8 + 3)  {
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

  ret &= test<int, ROW * COL>();
  ret &= test<float, ROW * COL>();

  return !(ret == true);
}

