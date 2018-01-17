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

#define Z (2)
#define Y (4)
#define X (16)

#define _DEBUG (0)

template<typename _Tp, size_t _Z, size_t _Y, size_t _X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_carray() {
  _Tp table[_Z * _Y * _X] { 0 };

  // generate array_view
  std::array_view<_Tp, 3> av(table, std::bounds<3>{_Z, _Y, _X});

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        int idx = i * _Y * _X + j * _X + k;
        if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
          // for items within for_each_n, the result value shall agree with the kernel
          if (table[idx] != i * 8 + j * 5 + k * 3 + 2)  {
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
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        std::cout << std::setw(5) << table[i * _Y * _X + j * _X + k];
      }
      std::cout << "\n";
    }
    std::cout << "=====\n\n";
  }
#endif

  return ret;
}

template<typename _Tp, size_t _Z, size_t _Y, size_t _X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_stdarray() {
  std::array<_Tp, _Z * _Y * _X> table { 0 };

  // generate array_view
  std::array_view<_Tp, 3> av(table.data(), std::bounds<3>{_Z, _Y, _X});

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        int idx = i * _Y * _X + j * _X + k;
        if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
          // for items within for_each_n, the result value shall agree with the kernel
          if (table[idx] != i * 8 + j * 5 + k * 3 + 2)  {
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
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        std::cout << std::setw(5) << table[i * _Y * _X + j * _X + k];
      }
      std::cout << "\n";
    }
    std::cout << "=====\n\n";
  }
#endif

  return ret;
}

template<typename _Tp, size_t _Z, size_t _Y, size_t _X, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test_vector() {
  std::vector<_Tp> table(_Z * _Y * _X, 0);

  // generate array_view
  std::array_view<_Tp, 3> av(table.data(), std::bounds<3>{_Z, _Y, _X});

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each_n(par, std::begin(av.bounds()) + FIRST_OFFSET, TEST_LENGTH, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        int idx = i * _Y * _X + j * _X + k;
        if ((idx >= FIRST_OFFSET) && idx < (FIRST_OFFSET + TEST_LENGTH)) {
          // for items within for_each_n, the result value shall agree with the kernel
          if (table[idx] != i * 8 + j * 5 + k * 3 + 2)  {
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
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        std::cout << std::setw(5) << table[i * _Y * _X + j * _X + k];
      }
      std::cout << "\n";
    }
    std::cout << "=====\n\n";
  }
#endif

  return ret;
}

int main() {
  bool ret = true;

  ret &= test_carray<int, Z, Y, X, 0, 2>();
  ret &= test_carray<float, Z, Y, X, 0, 2>();

  ret &= test_carray<int, Z, Y, X, X, X * 2>();
  ret &= test_carray<float, Z, Y, X, X, X * 2>();

  ret &= test_carray<int, Z, Y, X, X * 2 + X / 2, X / 2>();
  ret &= test_carray<float, Z, Y, X, X * 2 + X / 2, X / 2>();

  ret &= test_stdarray<int, Z, Y, X, 0, 2>();
  ret &= test_stdarray<float, Z, Y, X, 0, 2>();

  ret &= test_stdarray<int, Z, Y, X, X, X * 2>();
  ret &= test_stdarray<float, Z, Y, X, X, X * 2>();

  ret &= test_stdarray<int, Z, Y, X, X * 2 + X / 2, X / 2>();
  ret &= test_stdarray<float, Z, Y, X, X * 2 + X / 2, X / 2>();

  ret &= test_vector<int, Z, Y, X, 0, 2>();
  ret &= test_vector<float, Z, Y, X, 0, 2>();

  ret &= test_vector<int, Z, Y, X, X, X * 2>();
  ret &= test_vector<float, Z, Y, X, X, X * 2>();

  ret &= test_vector<int, Z, Y, X, X * 2 + X / 2, X / 2>();
  ret &= test_vector<float, Z, Y, X, X * 2 + X / 2, X / 2>();

  return !(ret == true);
}

