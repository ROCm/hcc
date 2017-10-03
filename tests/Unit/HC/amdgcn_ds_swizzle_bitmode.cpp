
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

  static_assert((pattern & 0x00008000) == 0, "The pattern provided is not in BitMode and therefore incompatible with this test!");
  short and_mask = pattern & 0x1F;
  short or_mask = (pattern >> 5) & 0x1F;
  short xor_mask = (pattern >> 10) & 0x1F;
  for(int j = 0; j < num; j+= 32) {
    for (unsigned int i = 0; i < 32; i++) {
      unsigned int k = (((i & and_mask) | or_mask) ^ xor_mask);
      expected[j+i] = av_input_x[j+k];
    }
  }

  return std::equal(expected.begin(), expected.end(), actual.begin());
}


int main() {
  bool pass = true;
  pass &= run_test<int,(short)0x5D69>(1024*1024);
  pass &= run_test<int,(short)0x5355>(1024*1024);

  pass &= run_test<unsigned int,(short)0x5D69>(1024*1024);
  pass &= run_test<unsigned int,(short)0x5355>(1024*1024);

  pass &= run_test<float,(short)0x5D69>(1024*1024);
  pass &= run_test<float,(short)0x5355>(1024*1024);

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
