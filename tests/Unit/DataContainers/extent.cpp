// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <amp.h>
#include <gtest/gtest.h>

#define N0 10
#define N1 10
#define N2 10

TEST(ClassExtent, Extent1D) {
  int n0 = N0;
  Concurrency::extent<1> ext(n0);

  EXPECT_EQ(n0, ext[0]);
}

TEST(ClassExtent, Extent2D) {
  int n0 = N0;
  int n1 = N1;
  Concurrency::extent<2> ext(n0, n1);

  EXPECT_EQ(n0, ext[0]);
  EXPECT_EQ(n1, ext[1]);
}

TEST(ClassExtent, Extent3D) {
  int n0 = N0;
  int n1 = N1;
  int n2 = N2;
  Concurrency::extent<3> ext(n0, n1, n2);

  EXPECT_EQ(n0, ext[0]);
  EXPECT_EQ(n1, ext[1]);
  EXPECT_EQ(n2, ext[2]);
}
