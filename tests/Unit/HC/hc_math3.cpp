
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <hc_math.hpp>

#include <algorithm>
#include <cmath>
#include <random>


// a test case which uses hc_math, which overrides math functions in the global namespace
// in this test case we check min / max specically
template<size_t GRID_SIZE, typename T, typename Q, typename R>
bool test() {
  using namespace hc;
  bool ret = true;

  array_view<T, 1> table1(GRID_SIZE); // input vector 1
  array_view<Q, 1> table2(GRID_SIZE); // input vector 2
  array_view<R, 1> table3(GRID_SIZE); // output vector calculated by GPU
  extent<1> ex(GRID_SIZE);

  // setup RNG
  std::random_device rd;
  std::default_random_engine gen(rd());

  // randomly produce input data
  typedef typename std::conditional<(sizeof(T) > sizeof(float)), double, float>::type  _RangeTypeT;
  std::uniform_real_distribution<_RangeTypeT> dis1(static_cast<_RangeTypeT>(1), static_cast<_RangeTypeT>(10));
  for (int i = 0; i < GRID_SIZE; ++i) table1[i] = dis1(gen);

  typedef typename std::conditional<(sizeof(Q) > sizeof(float)), double, float>::type  _RangeTypeQ;
  std::uniform_real_distribution<_RangeTypeQ> dis2(static_cast<_RangeTypeQ>(1), static_cast<_RangeTypeQ>(10));
  for (int i = 0; i < GRID_SIZE; ++i) table2[i] = dis2(gen);

#define TEST(func) \
  { \
    parallel_for_each(ex, [=](index<1>& idx) __HC__ { \
      table3(idx) = func(table1(idx), table2(idx)); \
    }); \
    accelerator().get_default_view().wait(); \
    int error = 0; \
    for (size_t i = 0; i < GRID_SIZE; ++i) { \
      R actual = table3[i];\
      R expected = std::func(table1[i],table2[i]);\
      R delta = std::fabs(actual - expected); \
      if (delta > expected * 0.0001) error++; \
    } \
    ret &= (error == 0); \
  }


  TEST(pow)

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<16, float,float,float>();
  ret &= test<16, int,float,float>();
  //ret &= test<16, float,int,float>(); // FIXME: skip this test as it breaks AMDGPU backend as of now
  ret &= test<16, int,int,float>();
  ret &= test<16, double,double,double>();
  ret &= test<16, int,double,double>();
  ret &= test<16, double,int,double>();
  ret &= test<16, int,int,double>();

  ret &= test<4096, float,float,float>();
  ret &= test<4096, int,float,float>();
  //ret &= test<4096, float,int,float>(); // FIXME: skip this test as it breaks AMDGPU backend as of now
  ret &= test<4096, int,int,float>();
  ret &= test<4096, double,double,double>();
  ret &= test<4096, int,double,double>();
  ret &= test<4096, double,int,double>();
  ret &= test<4096, int,int,double>();

  return !(ret == true);
}

