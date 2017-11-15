
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


// negative test
// no fill_n shall commence
template<typename T, size_t SIZE, int FIRST_OFFSET, int LAST_OFFSET>
bool test_negative(void) {

  using std::experimental::parallel::par;

  bool ret = true;
  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([](stdArray &input1,
                                               stdArray &input2) {
    std::fill_n(std::begin(input1) + FIRST_OFFSET, (LAST_OFFSET - FIRST_OFFSET), SIZE + 1);
    std::experimental::parallel::
    fill_n(par, std::begin(input2) + FIRST_OFFSET, (LAST_OFFSET - FIRST_OFFSET), SIZE + 1);
  });

  return ret;
}

// postive test
// fill_n shall commence
template<typename T, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test(void) {

  using std::experimental::parallel::par;

  bool ret = true;
  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([](stdArray &input1,
                                               stdArray &input2) {
    std::fill_n(std::begin(input1) + FIRST_OFFSET, TEST_LENGTH, SIZE + 1);
    std::experimental::parallel::
    fill_n(par, std::begin(input2) + FIRST_OFFSET, TEST_LENGTH, SIZE + 1);
  });

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

