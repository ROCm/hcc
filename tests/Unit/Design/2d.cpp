//_view RUN: %gtest_amp %s -o %t.out && %t.out

#include <hc.hpp>
#include <stdlib.h>
#include <iostream>
#ifndef __KALMAR_ACCELERATOR__
#include <gtest/gtest.h>
#endif

class myVecAdd {
 public:
  // CPU-side constructor. Written by the user
  myVecAdd(hc::array_view<int, 2>& a,
    hc::array_view<int, 2> &b,
    hc::array_view<int, 2> &c):
    a_(a), b_(b), c_(c) {
  }
  void operator() (hc::index<2> idx) [[hc]] {
    c_[idx] = a_[idx]+b_[idx];
  }
  void operator() (hc::tiled_index<2> idx) [[hc]] {
    c_[idx] = a_[idx]+b_[idx];
  }
 private:
  hc::array_view<int, 2> &c_;
  hc::array_view<int, 2> a_, b_;
};
void bar(void) [[cpu, hc]] {
  int* foo = reinterpret_cast<int*>(&myVecAdd::__cxxamp_trampoline);
}
#ifndef __KALMAR_ACCELERATOR__
#define M 20
#define N 40
TEST(Design, Final) {
  std::vector<int> vector_a(M*N),
                   vector_b(M*N);
  for (int i = 0; i < M*N; i++) {
    vector_a[i] = 100.0f * rand() / RAND_MAX;
    vector_b[i] = 100.0f * rand() / RAND_MAX;
  }
  hc::extent<2> e(M, N);
  hc::array_view<int, 2> av(e, vector_a);
  EXPECT_EQ(vector_a[2], av(0,2));
  hc::array_view<int, 2> bv(e, vector_b);
  { // Test untiled version
    hc::array_view<int, 2> c(e);
    myVecAdd mf(av, bv, c);
    hc::parallel_for_each(e, mf);
    int error=0;
    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
	std::cout << "av[" <<i<<","<<j<<"] = "<<av(i,j)<<"\n";
	std::cout << "bv[" <<i<<","<<j<<"] = "<<bv(i,j)<<"\n";
	std::cout << "c[" <<i<<","<<j<<"] = "<<c(i,j)<<"\n";
	error += abs(c(i, j) - (av(i, j) + bv(i, j)));
      }
    }
    EXPECT_EQ(0, error);
  }
  {
   // Test tiled version
    hc::array_view<int, 2> c(e);
    myVecAdd mf(av, bv, c);
    hc::parallel_for_each(e.tile(4, 4), mf);
    int error=0;
    for(int i = 0; i < M; i++) {
      for(int j = 0; j < N; j++) {
	std::cout << "av[" <<i<<","<<j<<"] = "<<av(i,j)<<"\n";
	std::cout << "bv[" <<i<<","<<j<<"] = "<<bv(i,j)<<"\n";
	std::cout << "c[" <<i<<","<<j<<"] = "<<c(i,j)<<"\n";
	error += abs(c(i, j) - (av(i, j) + bv(i, j)));
      }
    }
    EXPECT_EQ(0, error);
  }
}
#endif
