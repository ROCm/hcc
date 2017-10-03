
// RUN: %hc %s -o %t.out && %t.out

// Parallel STL headers
#include <coordinate>
#include <experimental/algorithm>
#include <experimental/numeric>
#include <experimental/execution_policy>

#define _DEBUG (0)
#include "test_base.h"


template<typename T, size_t SIZE>
bool test(void) {
  auto pred1 = [](const T& v) [[hc,cpu]] { return static_cast<int>(v) % 3 == 0; };
  auto pred2 = [](const T& v) [[hc,cpu]] { return static_cast<int>(v) == SIZE + 1; };
  auto pred3 = [](const T& v) [[hc,cpu]] { return v >= 0; };

  using std::experimental::parallel::par;

  bool ret = true;
  bool eq = true;
  // std::array
  typedef std::array<T, SIZE> stdArray;

  // any_of
  ret &= run_and_compare<T, SIZE, stdArray>([pred1, &eq](stdArray &input1,
                                                         stdArray &input2) {
    auto expected = std::any_of(std::begin(input1), std::end(input1), pred1);
    auto result   = std::experimental::parallel::
                    any_of(par, std::begin(input2), std::end(input2), pred1);
    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // none_of
  ret &= run_and_compare<T, SIZE, stdArray>([pred2, &eq](stdArray &input1,
                                                         stdArray &input2) {
    auto expected = std::none_of(std::begin(input1), std::end(input1), pred2);
    auto result   = std::experimental::parallel::
                    none_of(par, std::begin(input2), std::end(input2), pred2);
    eq = EQ(expected, result);
  }, false);
  ret &= eq;


  // all_of
  ret &= run_and_compare<T, SIZE, stdArray>([pred3, &eq](stdArray &input1,
                                                         stdArray &input2) {
    auto expected = std::all_of(std::begin(input1), std::end(input1), pred3);
    auto result   = std::experimental::parallel::
                    all_of(par, std::begin(input2), std::end(input2), pred3);
    eq = EQ(expected, result);
  }, false);
  ret &= eq;

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

