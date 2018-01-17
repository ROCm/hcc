
// RUN: %hc %s -o %t.out && %t.out

#include <typeinfo>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

#include <hc.hpp>

//#define DEBUG 2

#ifdef DEBUG
  #define DEBUG_MSG(MSG,...)  fprintf(stderr,"%s:%d ", __FILE__,__LINE__); fprintf(stderr, MSG, __VA_ARGS__); fprintf(stderr, "\n");
#else
  #define DEBUG_MSG(MSG,...)   
#endif


constexpr unsigned int WAVE_SIZE(64);

template <typename T>
unsigned int run_test(const int num, const int tile, const int iter) {

  std::vector<T> input(num);
  int i = 1234;
  std::generate(input.begin(), input.end(), [&]() {
    return i++;
  });
  hc::array_view<T,1> av_input(num, input);

  std::vector<T> actual(num);
  hc::array_view<T,1> av_actual(num, actual);
  hc::parallel_for_each(av_input.get_extent().tile(tile), 
                        [=](hc::tiled_index<1> idx) [[hc]] {
    T v = av_input[idx.global[0]];
    for (int i = 0; i < iter; i++) {
      v = hc::__amdgcn_wave_rr1(v);
    }
    av_actual[idx] = v;
  });
  av_actual.synchronize();
  
  std::vector<T> expected(num);
  auto r = expected.begin();
  for (int j = 0; j < num; j+=WAVE_SIZE) {
    for (int lane = 0; lane < WAVE_SIZE; lane++,r++) {
      *r = input[j + ((lane - iter)%WAVE_SIZE)];
    }
  }
  
  int index = 0;
  r = expected.begin();
  int errors = std::count_if(actual.begin(), actual.end(), [&](int i) {
    T expected = *r;
    r++;

#if DEBUG==2
    if (i!=expected) {
      std::cerr << "test failed(" << typeid(expected).name() <<  ", num=" << num  << ", tile=" << tile << ", iter=" << iter << ", bound_ctrl=" << bound_ctrl << ") ";
      std::cerr << "expected[" << index << "]=" << expected << "  actual[" << index << "]=" << *r << std::endl;
      assert(false);
    }
    index++;
#endif

    return i != expected; 
  });

  return errors;
}



template <typename T>
bool test() {
  bool pass = true;
  int errors;

  constexpr unsigned int GLOBAL_SIZE(4096);
  for (int iter = 1; iter <= 64; iter++) {
 
    errors = run_test<T>(GLOBAL_SIZE, 64, iter);
    DEBUG_MSG("%d errors",errors);
    pass &= (errors==0);
 
    errors = run_test<T>(GLOBAL_SIZE, 128, iter);
    DEBUG_MSG("%d errors",errors);
    pass &= (errors==0);
 
    errors = run_test<T>(GLOBAL_SIZE, 256, iter);
    DEBUG_MSG("%d errors",errors);
    pass &= (errors==0);
  }
  return pass;
}


int main() {
  bool pass = true;

  pass &= test<unsigned int>();
  pass &= test<int>();
  pass &= test<float>();

#ifdef DEBUG
  std::cout << (const char*)(pass?"passed!":"failed!") << std::endl;
#endif

  return !(pass == true);
}
