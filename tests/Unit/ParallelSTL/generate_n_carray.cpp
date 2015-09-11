// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

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

// negative test
// no generate_n shall commence
template<typename _Tp, size_t SIZE, int FIRST_OFFSET, int LAST_OFFSET>
bool test_negative() {

  // LAST_OFFSET is expected to be less than FIRST_OFFSET
  static_assert(LAST_OFFSET < FIRST_OFFSET, "test case is invalid!");

  _Tp table[SIZE] { 0 };

  // launch kernel with parallel STL generate_n
  using namespace std::experimental::parallel;
  auto iter = generate_n(par, std::begin(table) + FIRST_OFFSET, (LAST_OFFSET - FIRST_OFFSET), [] { return SIZE + 1; });

  // verify data
  bool ret = true;
  if (iter != std::begin(table) + FIRST_OFFSET) {
    ret = false;
  }
  for (int i = 0; i < SIZE; ++i) {
    // no transform shall commence
    if (table[i] != 0) {
      ret = false;
      break;
    }
  }
  return ret;
}

// positive test
// generate_n shall commence
template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test() {

  _Tp table[SIZE] { 0 };
  _Tp n { 0 };

  // initialize test data
  std::generate(std::begin(table), std::end(table), [&] { return n++; });

  // launch kernel with parallel STL generate_n
  using namespace std::experimental::parallel;
  auto iter = generate_n(par, std::begin(table) + FIRST_OFFSET, TEST_LENGTH, [] { return SIZE + 1; });

  // verify data
  bool ret = true;
  if (std::distance(std::begin(table) + FIRST_OFFSET, iter) != TEST_LENGTH) {
    ret = false;
  }
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

  // positive tests
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

  // negative tests
  ret &= test_negative<int, TEST_SIZE, 2, 0>();
  ret &= test_negative<unsigned, TEST_SIZE, 2, 0>();
  ret &= test_negative<float, TEST_SIZE, 2, 0>();
  ret &= test_negative<double, TEST_SIZE, 2, 0>();

  ret &= test_negative<int, TEST_SIZE, COL * 2, COL>();
  ret &= test_negative<unsigned, TEST_SIZE, COL * 2, COL>();
  ret &= test_negative<float, TEST_SIZE, COL * 2, COL>();
  ret &= test_negative<double, TEST_SIZE, COL * 2, COL>();

  ret &= test_negative<int, TEST_SIZE, COL * 2, COL * 2 - COL / 2>();
  ret &= test_negative<unsigned, TEST_SIZE, COL * 2, COL * 2 - COL / 2>();
  ret &= test_negative<float, TEST_SIZE, COL * 2, COL * 2 - COL / 2>();
  ret &= test_negative<double, TEST_SIZE, COL * 2, COL * 2 - COL / 2>();

  return !(ret == true);
}

