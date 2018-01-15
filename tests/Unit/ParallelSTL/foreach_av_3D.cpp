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

template<typename _Tp, size_t _Z, size_t _Y, size_t _X>
bool test_carray() {
  _Tp table[_Z * _Y * _X] { 0 };

  // generate array_view
  std::array_view<_Tp, 3> av(table, std::bounds<3>{_Z, _Y, _X});

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        if (table[i * _Y * _X + j * _X + k] != i * 8 + j * 5 + k * 3 + 2) {
          ret = false;
          break;
        }
      }
    }
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; j < _X; ++k) {
        std::cout << std::setw(5) << table[i * _Y * _X + j * _X + k];
      }
      std::cout << "\n";
    }
    std::cout << "=====\n\n";
  } 
#endif

  return ret;
}

template<typename _Tp, size_t _Z, size_t _Y, size_t _X>
bool test_stdarray() {
  std::array<_Tp, _Z * _Y * _X> table { 0 };

  // generate array_view
  std::array_view<_Tp, 3> av(table.data(), std::bounds<3>{_Z, _Y, _X});

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        if (table[i * _Y * _X + j * _X + k] != i * 8 + j * 5 + k * 3 + 2) {
          ret = false;
          break;
        }
      }
    }
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; j < _X; ++k) {
        std::cout << std::setw(5) << table[i * _Y * _X + j * _X + k];
      }
      std::cout << "\n";
    }
    std::cout << "=====\n\n";
  } 
#endif

  return ret;
}

template<typename _Tp, size_t _Z, size_t _Y, size_t _X>
bool test_vector() {
  std::vector<_Tp> table(_Z * _Y * _X, 0);

  // generate array_view
  std::array_view<_Tp, 3> av(table.data(), std::bounds<3>{_Z, _Y, _X});

  // range for
  for (auto idx : av.bounds()) {
    av[idx] = 1;
  }

  // test kernel
  auto f = [&](const std::offset<3>& idx) {
    av[idx] = idx[0] * 8 + idx[1] * 5 + idx[2] * 3 + 2;
  };

  // get iterator of array_view
  auto first = std::begin(av.bounds());
  auto last = std::end(av.bounds());

  // launch kernel with parallel STL for_each
  using namespace std::experimental::parallel;
  for_each(par, first, last, f);

  // verify data
  bool ret = true;
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; k < _X; ++k) {
        if (table[i * _Y * _X + j * _X + k] != i * 8 + j * 5 + k * 3 + 2) {
          ret = false;
          break;
        }
      }
    }
  }

#if _DEBUG
  for (int i = 0; i < _Z; ++i) {
    for (int j = 0; j < _Y; ++j) {
      for (int k = 0; j < _X; ++k) {
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

  ret &= test_carray<int, Z, Y, X>();
  ret &= test_carray<float, Z, Y, X>();

  ret &= test_stdarray<int, Z, Y, X>();
  ret &= test_stdarray<float, Z, Y, X>();

  ret &= test_vector<int, Z, Y, X>();
  ret &= test_vector<float, Z, Y, X>();

  return !(ret == true);
}

