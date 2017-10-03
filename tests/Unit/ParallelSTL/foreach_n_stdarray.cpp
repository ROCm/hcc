
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

template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test() {

  std::array<_Tp, SIZE> table { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // test kernel
  auto f = [&](_Tp& v) [[hc,cpu]]
  {
    v *= 8;
    v += 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(table) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if ((i >= FIRST_OFFSET) && i < (FIRST_OFFSET + TEST_LENGTH)) {
      // for items within for_each_n, the result value shall agree with the kernel
      if (table[i] != i * 8 + 3)  {
        ret = false;
        break;
      }
    } else {
      // for items outside for_each_n, the result value shall be the initial value
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

  ret &= test<int, ROW * COL, 0, 2>();
  ret &= test<float, ROW * COL, 0, 2>();

  ret &= test<int, ROW * COL, COL, COL * 2>();
  ret &= test<float, ROW * COL, COL, COL * 2>();

  ret &= test<int, ROW * COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test<float, ROW * COL, COL * 2 + COL / 2, COL / 2>();

  return !(ret == true);
}

