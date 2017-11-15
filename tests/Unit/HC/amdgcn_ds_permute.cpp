
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



template <typename T>
bool run_test(const int num) {

  std::vector<T> input_x(num);
  std::vector<int> input_y(num);

  // initialize the input data
  std::default_random_engine random_gen;

  T max = std::is_signed<T>::value ? INT_MAX : UINT_MAX;
  T min = std::is_signed<T>::value ? INT_MIN : 0;

  typedef typename std::conditional<std::is_signed<T>::value, int, unsigned int>::type _RangeType;
  std::uniform_int_distribution<_RangeType> distribution(static_cast<_RangeType>(min), static_cast<_RangeType>(max));
  auto gen = std::bind(distribution, random_gen);
  std::generate(input_x.begin(), input_x.end(), gen);

  constexpr int wave_size = 64;

  // generate the lane IDs
  int counter = 0;
  std::generate(input_y.begin(), input_y.end(), [&]() {
    return counter++ % wave_size;
  });
  for (auto wave_start = input_y.begin(); wave_start != input_y.end(); wave_start+=wave_size) {
    std::random_shuffle(wave_start, wave_start + wave_size);
  }

  hc::array_view<T,1> av_input_x(num, input_x);
  hc::array_view<int,1> av_input_y(num, input_y);

  std::vector<T> actual(num);
  hc::array_view<T,1> av_actual(num, actual);
  hc::parallel_for_each(av_input_x.get_extent(),
                        [=](hc::index<1> idx) [[hc]] {
    av_actual[idx] = hc::__amdgcn_ds_permute(av_input_y[idx]<<2, av_input_x[idx]);
  });
  av_actual.synchronize();

  std::vector<T> expected(num);
  for(int j = 0; j < num; j+= wave_size) {
    for (int i = 0; i < wave_size; i++) {
      expected[ j+input_y[j+i] ] = input_x[j+i];
    }
  }

  return std::equal(expected.begin(), expected.end(), actual.begin());
}


int main() {
  bool pass = true;

  pass &= run_test<unsigned int>(1024*1024);
  pass &= run_test<int>(1024*1024);
  pass &= run_test<float>(1024*1024);

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
