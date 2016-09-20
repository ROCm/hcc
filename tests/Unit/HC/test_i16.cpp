
// RUN: %hc %s -o %t.out && %t.out

// a test to check I16 type can be used in HCC

#include <hc.hpp>
#include <cstdio>


int main() {
  short a = 1234;
  short b = 5678;

  hc::array_view<short,1> av_1(1);
  av_1[0] = a;

  hc::array_view<short,1> av_3(1);
  av_3[0] = b;

  hc::array_view<short,1> av_2(1);


  hc::parallel_for_each(av_1.get_extent(), [=](hc::index<1> i) [[hc]] {
    av_1[i] += av_3[i];
  }).wait();

  hc::parallel_for_each(av_1.get_extent(), [=](hc::index<1> i) [[hc]] {
    av_2[i] = (short)av_1[i];
  });

  printf("av_1: %d\n",(short)av_2[0]);

  return 0;
}
