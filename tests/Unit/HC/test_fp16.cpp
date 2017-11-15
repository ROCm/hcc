
// RUN: %hc %s -o %t.out && %t.out

// a test to check FP16 type can be used in HCC

#include <hc.hpp>
#include <cstdio>

typedef __fp16 hcc_fp16;

int main() {
  hcc_fp16 a = 1234.0;
  hcc_fp16 b = 5678.0;

  hc::array_view<__fp16,1> av_1(1);
  av_1[0] = a;

  hc::array_view<__fp16,1> av_3(1);
  av_3[0] = b;

  hc::array_view<float,1> av_2(1);
  
  
  hc::parallel_for_each(av_1.get_extent(), [=](hc::index<1> i) [[hc]] {
    av_1[i] += av_3[i];
  }).wait();
    
  hc::parallel_for_each(av_1.get_extent(), [=](hc::index<1> i) [[hc]] {
    av_2[i] = (float)av_1[i];
  });

  printf("av_1: %f\n",(float)av_2[0]);

  return 0;
}
