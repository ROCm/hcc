// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <climits>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <type_traits>


#include <hc.hpp>

//#define DEBUG 1

#ifdef DEBUG
  #define DEBUG_MSG(MSG,...)  fprintf(stderr,"%s:%d ", __FILE__,__LINE__); fprintf(stderr, MSG, __VA_ARGS__); fprintf(stderr, "\n");
#else
  #define DEBUG_MSG(MSG,...)   
#endif



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

  constexpr int wave_size = 64;

  // generate the lane IDs
  int counter = 0;
  std::generate(input_y.begin(), input_y.end(), [&]() {
    counter = (counter == wave_size)?0:counter++;
    return counter;
  });
  for (auto wave_start = input_y.begin(); wave_start != input_y.end(); wave_start+=wave_size) {
    std::random_shuffle(wave_start, wave_start + wave_size);
  }

  hc::array_view<T,1> av_input_x(num, input_x);
  hc::array_view<T,1> av_input_y(num, input_y);

  std::vector<T> actual(num);
  hc::array_view<T,1> av_actual(num, actual);
  hc::parallel_for_each(av_input_x.get_extent(), 
                        [=](hc::index<1> idx) [[hc]] {
    av_actual[idx] = hc::__amdgcn_ds_permute(av_input_y[idx], av_input_x[idx]);
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

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
