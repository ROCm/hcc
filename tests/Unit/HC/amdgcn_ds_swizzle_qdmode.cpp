
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

//#define DEBUG 1

#ifdef DEBUG
  #define DEBUG_MSG(MSG,...)  fprintf(stderr,"%s:%d ", __FILE__,__LINE__); fprintf(stderr, MSG, __VA_ARGS__); fprintf(stderr, "\n");
#else
  #define DEBUG_MSG(MSG,...)
#endif



template <typename T, short pattern>
bool run_test(const int num) {

  std::vector<T> input_x(num);

  // initialize the input data
  std::default_random_engine random_gen;

  T max = std::is_signed<T>::value ? INT_MAX : UINT_MAX;
  T min = std::is_signed<T>::value ? INT_MIN : 0;

  typedef typename std::conditional<std::is_signed<T>::value, int, unsigned int>::type _RangeType;
  std::uniform_int_distribution<_RangeType> distribution(static_cast<_RangeType>(min), static_cast<_RangeType>(max));
  auto gen = std::bind(distribution, random_gen);
  std::generate(input_x.begin(), input_x.end(), gen);

  hc::array_view<T,1> av_input_x(num, input_x);

  std::vector<T> actual(num);
  hc::array_view<T,1> av_actual(num, actual);
  hc::parallel_for_each(av_input_x.get_extent(),
                        [=](hc::index<1> idx) [[hc]] {
    av_actual[idx] = hc::__amdgcn_ds_swizzle(av_input_x[idx],pattern);
  });
  av_actual.synchronize();

  std::vector<T> expected(num);

  static_assert((pattern & 0x00008000) != 0, "The pattern provided is not in QDMode and therefore incompatible with this test!");
  std::vector<int> patterns(4);
  patterns[0] = pattern & 0x03;
  patterns[1] = (pattern >> 2) & 0x03;
  patterns[2] = (pattern >> 4) & 0x03;
  patterns[3] = (pattern >> 6) & 0x03;
  for(int j = 0; j < num; j+= 4) {
    for (int i = 0; i < 4; i++) {
      expected[j+i] = input_x[j+patterns[i]];
    }
  }

  return std::equal(expected.begin(), expected.end(), actual.begin());
}


int main() {
  bool pass = true;
  pass &= run_test<int,(short)0x8055>(1024*1024);
  pass &= run_test<int,(short)0x80E4>(1024*1024);
  pass &= run_test<int,(short)0x80FF>(1024*1024);

  pass &= run_test<unsigned int,(short)0x8055>(1024*1024);
  pass &= run_test<unsigned int,(short)0x80E4>(1024*1024);
  pass &= run_test<unsigned int,(short)0x80FF>(1024*1024);

  pass &= run_test<float,(short)0x8055>(1024*1024);
  pass &= run_test<float,(short)0x80E4>(1024*1024);
  pass &= run_test<float,(short)0x80FF>(1024*1024);

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
