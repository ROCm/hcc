// RUN: %gtest_amp %s -o %t.out && %t.out

#include <hc/hc.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>

class myVecAdd {
 public:
  // CPU-side constructor. Written by the user
  myVecAdd(hc::array_view<int>& a,
    hc::array_view<int> &b,
    hc::array_view<int> &c):
    a_(a), b_(b), c_(c) {
  }
  void operator() (hc::index<1> idx) const [[hc]] {
    c_[idx] = a_[idx]+b_[idx];
  }
 private:
  hc::array_view<int> a_, b_, c_;
};
TEST(Design, Final) {
  const int vecSize = 100;

  // Alloc & init input data
  hc::extent<1> e(vecSize);
  hc::array<int, 1> a(vecSize);
  hc::array<int, 1> b(vecSize);
  hc::array<int, 1> c(vecSize);
  int sum = 0;


  hc::array_view<int> ga(a);
  hc::array_view<int> gb(b);
  hc::array_view<int> gc(c);
  for (hc::index<1> i(0); i[0] < vecSize; i++) {
    ga[i] = 100.0f * rand() / RAND_MAX;
    gb[i] = 100.0f * rand() / RAND_MAX;
    sum += ga[i] + gb[i];
  }
  myVecAdd mf(ga, gb, gc);
  hc::parallel_for_each(e, mf);

  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += gc[i] - (ga[i] + gb[i]);
  }
  EXPECT_EQ(error, 0);
}