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
#include <numeric>
#include <random>

#define ROW (8)
#define COL (16)
#define TEST_SIZE (ROW * COL)

#define _DEBUG (0)

template<typename _Tp, size_t SIZE>
bool test() {

  _Tp input[SIZE] { 0 };
  _Tp output[SIZE] { 0 };

  // initialize test data
  std::iota(std::begin(input), std::end(input), 0);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_int_distribution<int> dis(0, SIZE);

  // randomly pick one element from the input array
  int indexRandom = dis(gen);

  // launch kernel with parallel STL copy
  using namespace std::experimental::parallel;
  auto iter = replace_copy(par, std::begin(input), std::end(input), std::begin(output), input[indexRandom], static_cast<_Tp>(SIZE + 1));

  // verify data
  bool ret = true;
  if (iter != std::begin(output) + SIZE) {
    ret = false;
  }
  for (int i = 0; i < SIZE; ++i) {
    if (i != indexRandom) {
      if (output[i] != i)  {
        ret = false;
        break;
      }
    } else {
      if (output[i] != SIZE + 1) {
        ret = false;
        break;
      }
    }
  }

#if _DEBUG 
  for (int i = 0; i < ROW; ++i) {
    for (int j = 0; j < COL; ++j) {
      std::cout << std::setw(5) << output[i * COL + j];
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

