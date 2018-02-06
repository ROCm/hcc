
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/execution_policy>


#define _DEBUG (0)
#include "test_base.h"


// negative test
// no generate_n shall commence
template<typename T, size_t SIZE, int FIRST_OFFSET, int LAST_OFFSET>
bool test_negative(void) {

  auto f = []() [[hc,cpu]] { return SIZE + 1; };
  using std::experimental::parallel::par;

  bool ret = true;

  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([f](stdArray &input1,
                                                stdArray &input2) {
    std::generate_n(std::begin(input1) + FIRST_OFFSET, (LAST_OFFSET - FIRST_OFFSET), f);
    std::experimental::parallel::
    generate_n(par, std::begin(input2) + FIRST_OFFSET, (LAST_OFFSET - FIRST_OFFSET), f);
  });

  return ret;
}

// positive test
// generate_n shall commence
template<typename T, size_t SIZE, size_t FIRST_OFFSET, size_t TEST_LENGTH>
bool test(void) {

  auto f = []() [[hc,cpu]] { return SIZE + 1; };
  using std::experimental::parallel::par;

  bool ret = true;

  // std::array
  typedef std::array<T, SIZE> stdArray;
  ret &= run_and_compare<T, SIZE, stdArray>([f](stdArray &input1,
                                                stdArray &input2) {
    std::generate(std::begin(input1) + FIRST_OFFSET, std::begin(input1) + TEST_LENGTH, f);
    std::experimental::parallel::
    generate_n(par, std::begin(input2) + FIRST_OFFSET, TEST_LENGTH, f);
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

  // negative tests
  ret &= test_negative<int, TEST_SIZE, 2, 0>();
  ret &= test_negative<unsigned, TEST_SIZE, 2, 0>();
  ret &= test_negative<float, TEST_SIZE, 2, 0>();
  ret &= test_negative<double, TEST_SIZE, 2, 0>();

  return !(ret == true);
}

