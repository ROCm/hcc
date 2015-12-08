// XFAIL: Linux,boltzmann
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <random>

// a test case which uses hc_math, which overrides math functions in the global namespace
// in this test case we check min / max specically
template<size_t GRID_SIZE, typename T>
bool test() {
  using namespace hc;
  bool ret = true;

  T table1[GRID_SIZE]; // input vector 1
  T table2[GRID_SIZE]; // input vector 2
  T table3[GRID_SIZE]; // output vector calculated by GPU
  extent<1> ex(GRID_SIZE);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<T> dis(1, GRID_SIZE);

  // randomly produce input data
  std::for_each(std::begin(table1), std::end(table1), [&](T& v) { v = dis(gen); });
  std::for_each(std::begin(table2), std::end(table2), [&](T& v) { v = dis(gen); });

#define TEST(func) \
  { \
    parallel_for_each(ex, [&](index<1>& idx) __HC__ { \
      table3[idx[0]] = func(table1[idx[0]], table2[idx[0]]); \
    }); \
    accelerator().get_default_view().wait(); \
    for (size_t i = 0; i < GRID_SIZE; ++i) { \
      if (func(table1[i], table2[i]) != table3[i]) { \
        ret = false; \
        break; \
      } \
    } \
  } 

  TEST(min)
  TEST(max)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<16, float>();
  ret &= test<16, double>();
  ret &= test<4096, float>();
  ret &= test<4096, double>();

  return !(ret == true);
}

