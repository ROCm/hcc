
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


unsigned int convert_32_to_24(unsigned int x) {
  return x & 0x00FFFFFF;
}

int convert_32_to_24(int x) {
  return (x << 8) >> 8;
}


template <typename T>
bool run_test(const int num) {

  std::vector<T> input_x(num);
  std::vector<T> input_y(num);

  // initialize the input data
  std::default_random_engine random_gen;

  T max = std::is_signed<T>::value ? INT_MAX : UINT_MAX;
  T min = std::is_signed<T>::value ? INT_MIN : 0;

  std::uniform_int_distribution<T> distribution(min, max);
  auto gen = std::bind(distribution, random_gen);
  std::generate(input_x.begin(), input_x.end(), gen);
  std::generate(input_y.begin(), input_y.end(), gen);

  hc::array_view<T,1> av_input_x(num, input_x);
  hc::array_view<T,1> av_input_y(num, input_y);


  std::vector<T> actual(num);
  hc::array_view<T,1> av_actual(num, actual);
  hc::parallel_for_each(av_input_x.get_extent(),
                        [=](hc::index<1> idx) [[hc]] {
    av_actual[idx] = hc::__mul24(av_input_x[idx], av_input_y[idx]);
  });
  av_actual.synchronize();


  std::vector<T> expected(num);
  int i = 0;
  std::generate(expected.begin(), expected.end(), [&]() {
    T x = convert_32_to_24(input_x[i]);
    T y = convert_32_to_24(input_y[i]);
    i++;
    return x * y;
  });


  return std::equal(expected.begin(), expected.end(), actual.begin());
}


int main() {
  bool pass = true;

  pass &= run_test<unsigned int>(1024*1024);
  pass &= run_test<int>(1024*1024);

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
