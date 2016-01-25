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

template<typename _Tp, size_t Y, size_t X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_carray() {
  _Tp table[Y * X] { 0 };

  // generate array_view
  std::array_view<_Tp, 2> av(table, std::bounds<2>{Y, X});

  // test kernel
  auto f = [&](const std::offset<2>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < Y; ++i) {
    for (int j = 0; j < X; ++j) {
      int idx = i * X + j;
      if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
        // for items within for_each_n, the result value shall agree with the kernel
        if (table[idx] != i * 8 + j * 5 + 3)  {
          ret = false;
          break;
        }
      } else {
        // for items outside for_each_n, the result value shall be the initial value
        if (table[idx] != 0) {
          ret = false;
          break;
        }
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

template<typename _Tp, size_t Y, size_t X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_stdarray() {
  std::array<_Tp, Y * X> table { 0 };

  // generate array_view
  std::array_view<_Tp, 2> av(table.data(), std::bounds<2>{Y, X});

  // test kernel
  auto f = [&](const std::offset<2>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < Y; ++i) {
    for (int j = 0; j < X; ++j) {
      int idx = i * X + j;
      if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
        // for items within for_each_n, the result value shall agree with the kernel
        if (table[idx] != i * 8 + j * 5 + 3)  {
          ret = false;
          break;
        }
      } else {
        // for items outside for_each_n, the result value shall be the initial value
        if (table[idx] != 0) {
          ret = false;
          break;
        }
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

template<typename _Tp, size_t Y, size_t X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_vector() {
  std::vector<_Tp> table(Y * X, 0);

  // generate array_view
  std::array_view<_Tp, 2> av(table.data(), std::bounds<2>{Y, X});

  // test kernel
  auto f = [&](const std::offset<2>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + 3;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < Y; ++i) {
    for (int j = 0; j < X; ++j) {
      int idx = i * X + j;
      if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
        // for items within for_each_n, the result value shall agree with the kernel
        if (table[idx] != i * 8 + j * 5 + 3)  {
          ret = false;
          break;
        }
      } else {
        // for items outside for_each_n, the result value shall be the initial value
        if (table[idx] != 0) {
          ret = false;
          break;
        }
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

  ret &= test_carray<int, ROW, COL, 0, 2>();
  ret &= test_carray<float, ROW, COL, 0, 2>();

  ret &= test_carray<int, ROW, COL, COL, COL * 2>();
  ret &= test_carray<float, ROW, COL, COL, COL * 2>();

  ret &= test_carray<int, ROW, COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_carray<float, ROW, COL, COL * 2 + COL / 2, COL / 2>();

  ret &= test_stdarray<int, ROW, COL, 0, 2>();
  ret &= test_stdarray<float, ROW, COL, 0, 2>();

  ret &= test_stdarray<int, ROW, COL, COL, COL * 2>();
  ret &= test_stdarray<float, ROW, COL, COL, COL * 2>();

  ret &= test_stdarray<int, ROW, COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_stdarray<float, ROW, COL, COL * 2 + COL / 2, COL / 2>();

  ret &= test_vector<int, ROW, COL, 0, 2>();
  ret &= test_vector<float, ROW, COL, 0, 2>();

  ret &= test_vector<int, ROW, COL, COL, COL * 2>();
  ret &= test_vector<float, ROW, COL, COL, COL * 2>();

  ret &= test_vector<int, ROW, COL, COL * 2 + COL / 2, COL / 2>();
  ret &= test_vector<float, ROW, COL, COL * 2 + COL / 2, COL / 2>();

  return !(ret == true);
}

