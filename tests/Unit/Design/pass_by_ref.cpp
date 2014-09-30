// RUN: %gtest_amp %s -o %t.out 
// RUN: %t.out

#include <amp.h>
#include <stdlib.h>
#include <iostream>
#ifndef __GPU__
#include <gtest/gtest.h>
#endif

class myVecAdd {
 public:
  // CPU-side constructor. Written by the user
  myVecAdd(Concurrency::array_view<int>& a,
    Concurrency::array_view<int> &b,
    Concurrency::array<int, 1> &c):
    a_(a), b_(b), c_(c) {
  }
  void operator() (Concurrency::index<1> idx) restrict(amp) {
    c_[idx] = a_[idx]+b_[idx];
  }
 private:
  Concurrency::array_view<int> a_, b_;
  Concurrency::array<int>& c_;
};
void bar(void) restrict(amp,cpu) {
  int* foo = reinterpret_cast<int*>(&myVecAdd::__cxxamp_trampoline);
}
#ifndef __GPU__
TEST(Design, Final) {
  const int vecSize = 100;

  // Alloc & init input data
  Concurrency::extent<1> e(vecSize);
  Concurrency::array<int, 1> a(vecSize);
  Concurrency::array<int, 1> b(vecSize);
  Concurrency::array<int, 1> c(vecSize);
  int sum = 0;
  for (Concurrency::index<1> i(0); i[0] < vecSize; i++) {
    a[i] = 100.0f * rand() / RAND_MAX;
    b[i] = 100.0f * rand() / RAND_MAX;
    sum += a[i] + b[i];
  }

  Concurrency::array_view<int> ga(a);
  Concurrency::array_view<int> gb(b);
  myVecAdd mf(ga, gb, c);
  Concurrency::parallel_for_each(
    e,
    mf);

  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += c[Concurrency::index<1>(i)] - (ga[i] + gb[i]);
  }
  EXPECT_EQ(error, 0);
}
#endif
