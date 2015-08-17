// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

// C++ headers
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <iterator>

#define ROW (2)
#define COL (8)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {
  bool ret = true;

  _Tp table[SIZE] { _Tp{} };

  // initialize test data
  std::iota(std::begin(table), std::end(table), 1);
  _Tp buffer[SIZE] { _Tp{} };
  std::transform(std::begin(table), std::end(table), std::begin(buffer), std::negate<_Tp>());
  _Tp expected = std::accumulate(std::begin(buffer), std::end(buffer), _Tp{}, std::plus<_Tp>());

  // launch kernel with parallel STL transform reduce
  using namespace std::experimental::parallel;

  ret &= (expected == transform_reduce(std::begin(table), std::end(table), std::negate<_Tp>(), _Tp{}, std::plus<_Tp>()));
  ret &= (expected == transform_reduce(par, std::begin(table), std::end(table), std::negate<_Tp>(), _Tp{}, std::plus<_Tp>()));

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

