// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <iostream>

#define ERROR_THRESHOLD (1E-4)

//#define DEBUG 1

// a test case which uses hc_math, which overrides math functions in the global namespace
template<typename T, size_t GRID_SIZE>
bool test() {
  using namespace hc;
  bool ret = true;

  T table[GRID_SIZE];
  extent<1> ex(GRID_SIZE);

#ifdef DEBUG
#define REPORT_ERROR_IF(COND,F) if (COND) { std::cout << #F << " test failed!" << std::endl; }
#else
#define REPORT_ERROR_IF(COND,F)
#endif

#define TEST(func) \
  { \
    std::fill(std::begin(table), std::end(table), (T)(0)); \
    parallel_for_each(ex, [&](index<1>& idx) __HC__ { \
      table[idx[0]] = func((T)(idx[0]+1)); \
    }); \
    accelerator().get_default_view().wait(); \
    float error = 0.0f; \
    for (size_t i = 0; i < GRID_SIZE; ++i) { \
      T actual = table[i];\
      T expected = (T)func((T)(i+1));\
      float delta = fabs(actual - expected); \
      error+=delta;\
    } \
    REPORT_ERROR_IF(!(error<=ERROR_THRESHOLD),func);\
    ret &= (error <= ERROR_THRESHOLD); \
  } 


  TEST(sqrt)
  TEST(fabs)
  TEST(cbrt)
  TEST(log)
  TEST(ilogb)
  TEST(isnormal)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<int,16>();
  ret &= test<unsigned int,16>();
  ret &= test<float,16>();
  ret &= test<double,16>();

  return !(ret == true);
}

