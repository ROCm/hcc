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

template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // launch kernel with parallel STL generate_n
  using namespace std::experimental::parallel;
  generate_n(par, std::begin(table) + FIRST_OFFSET, TEST_LENGTH, [] { return SIZE + 1; });

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if ((i >= FIRST_OFFSET) && i < (FIRST_OFFSET + TEST_LENGTH)) {
      // for items within generate_n, the result value shall agree with the kernel
      if (table[i] != SIZE + 1)  {
        ret = false;
        break;
      }
    } else {
      // for items outside generate_n, the result value shall be the initial value
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

  ret &= test<int, TEST_SIZE, 0, 2>();
  ret &= test<unsigned, TEST_SIZE, 0, 2>();
  ret &= test<float, TEST_SIZE, 0, 2>();
  ret &= test<double, TEST_SIZE, 0, 2>();

  ret &= test<int, TEST_SIZE, COL, COL * 2>();
  ret &= test<unsigned, TEST_SIZE, COL, COL * 2>();
  ret &= test<float, TEST_SIZE, COL, COL * 2>();
  ret &= test<double, TEST_SIZE, COL, COL * 2>();

  ret &= test<int, TEST_SIZE, COL * 2 + COL / 2, COL / 2>();
  ret &= test<unsigned, TEST_SIZE, COL * 2 + COL / 2, COL / 2>();
  ret &= test<float, TEST_SIZE, COL * 2 + COL / 2, COL / 2>();
  ret &= test<double, TEST_SIZE, COL * 2 + COL / 2, COL / 2>();

  return !(ret == true);
}

