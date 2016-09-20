
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

  array_view<T, 1> table(GRID_SIZE);
  extent<1> ex(GRID_SIZE);

#ifdef DEBUG
#define REPORT_ERROR_IF(COND,F,E) if (COND) { std::cout << #F <<  " cumulative error=" << E << ", test failed!" << std::endl; }
#define REPORT_DELTA_IF(COND,F,ARG,EXP,ACT) if (COND) { std::cout << #F << "(" << ARG << ") expected="<< EXP << ", actual=" << ACT << std::endl; }
#else
#define REPORT_ERROR_IF(COND,F,E)
#define REPORT_DELTA_IF(COND,F,ARG,EXP,ACT) 
#endif

#define TEST(func) \
  { \
    for (int i = 0; i < GRID_SIZE; ++i) table[i] = T(); \
    parallel_for_each(ex, [=](index<1>& idx) __HC__ { \
      table(idx) = func((T)(idx[0]+1)); \
    }); \
    accelerator().get_default_view().wait(); \
    float error = 0.0f; \
    for (size_t i = 0; i < GRID_SIZE; ++i) { \
      T actual = table[i];\
      T expected = (T)func((T)(i+1));\
      double delta = fabs((double)actual - (double)expected); \
      REPORT_DELTA_IF(delta>=ERROR_THRESHOLD, func, (i+1), expected, actual);\
      error+=delta;\
    } \
    REPORT_ERROR_IF(!(error<=ERROR_THRESHOLD),func,error);\
    ret &= (error <= ERROR_THRESHOLD); \
  } 


  TEST(sqrt)
  TEST(fabs)
  TEST(cbrt)
  TEST(log)
  TEST(ilogb)
  TEST(isnormal)
  TEST(cospi)
  TEST(sinpi)
  TEST(rsqrt)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<int,16>();
  ret &= test<float,16>();
  ret &= test<double,16>();

  return !(ret == true);
}

