// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>

#define ERROR_THRESHOLD (1E-4)

// a test case which uses hc_math, which overrides math functions in the global namespace
template<size_t GRID_SIZE>
bool test() {
  using namespace hc;
  bool ret = true;

  float table[GRID_SIZE];
  extent<1> ex(GRID_SIZE);

#define TEST(func) \
  { \
    std::fill(std::begin(table), std::end(table), 0.0f); \
    parallel_for_each(ex, [&](index<1>& idx) __attribute((hc)) { \
      table[idx[0]] = func(float(idx[0])); \
    }); \
    accelerator().get_default_view().wait(); \
    float error = 0.0f; \
    for (size_t i = 0; i < GRID_SIZE; ++i) { \
      error += fabs(table[i] - func(float(i))); \
    } \
    ret &= (error <= ERROR_THRESHOLD); \
  } 

  TEST(sqrt)
  TEST(fabs)
  TEST(cbrt)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<16>();

  return !(ret == true);
}

