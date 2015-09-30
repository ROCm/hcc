// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

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

template<typename _Tp, size_t SIZE>
bool test_carray() {
  _Tp table[SIZE] { 0 };

  // generate array_view
  std::array_view<_Tp, 1> av(table, std::bounds<1>(SIZE));

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != i * 8 + 3) {
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

template<typename _Tp, size_t SIZE>
bool test_stdarray() {
  std::array<_Tp, SIZE> table { 0 };

  // generate array_view
  std::array_view<_Tp, 1> av(table.data(), std::bounds<1>(SIZE));

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != i * 8 + 3) {
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

template<typename _Tp, size_t SIZE>
bool test_vector() {
  std::vector<_Tp> table(SIZE, 0);

  // generate array_view
  std::array_view<_Tp, 1> av(table.data(), std::bounds<1>(SIZE));

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<1>& idx) {
    av[idx] = idx[0] * 8 + 3;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < SIZE; ++i) {
    if (table[i] != i * 8 + 3) {
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

  ret &= test_carray<int, ROW * COL>();
  ret &= test_carray<float, ROW * COL>();

  ret &= test_stdarray<int, ROW * COL>();
  ret &= test_stdarray<float, ROW * COL>();

  ret &= test_vector<int, ROW * COL>();
  ret &= test_vector<float, ROW * COL>();

  return !(ret == true);
}

