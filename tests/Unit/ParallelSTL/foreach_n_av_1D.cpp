// XFAIL: *
// RUN: %hc %s -o %t.out && %t.out

// FIXME: PSTL on std::array_view remains TBD

// Parallel STL headers
#include <array_view>
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <vector>

#define ROW (8)
#define COL (16)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_carray() {
  _Tp table[SIZE] { 0 };

  // generate array_view
  std::array_view<_Tp, 1> av(table, std::bounds<1>(SIZE));

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

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
      if (table[i] != 0) {
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

template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_stdarray() {
  std::array<_Tp, SIZE> table { 0 };

  // generate array_view
  std::array_view<_Tp, 1> av(table.data(), std::bounds<1>(SIZE));

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

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
      if (table[i] != 0) {
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

template<typename _Tp, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_vector() {
  std::vector<_Tp> table(SIZE, 0);

  // generate array_view
  std::array_view<_Tp, 1> av(table.data(), std::bounds<1>(SIZE));

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

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
      if (table[i] != 0) {
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

  ret &= test_carray<int, ROW * COL, 0, 2>();
  ret &= test_carray<float, ROW * COL, 0, 2>();

  ret &= test_carray<int, ROW * COL, COL, COL * 2>();
  ret &= test_carray<float, ROW * COL, COL, COL * 2>();

  ret &= test_carray<int, ROW * COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_carray<float, ROW * COL, COL * 2 + COL / 2, COL / 2>();

  ret &= test_stdarray<int, ROW * COL, 0, 2>();
  ret &= test_stdarray<float, ROW * COL, 0, 2>();

  ret &= test_stdarray<int, ROW * COL, COL, COL * 2>();
  ret &= test_stdarray<float, ROW * COL, COL, COL * 2>();

  ret &= test_stdarray<int, ROW * COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_stdarray<float, ROW * COL, COL * 2 + COL / 2, COL / 2>();

  ret &= test_vector<int, ROW * COL, 0, 2>();
  ret &= test_vector<float, ROW * COL, 0, 2>();

  ret &= test_vector<int, ROW * COL, COL, COL * 2>();
  ret &= test_vector<float, ROW * COL, COL, COL * 2>();

  ret &= test_vector<int, ROW * COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_vector<float, ROW * COL, COL * 2 + COL / 2, COL / 2>();

  return !(ret == true);
}

