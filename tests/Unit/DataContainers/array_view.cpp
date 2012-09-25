// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <stdlib.h>
#include <amp.h>
#include <gtest/gtest.h>

#define N0 5000

int init1D(std::vector<int>& vec) {
  int n = N0;
  for (int i = 0; i < n; ++i) {
    vec.push_back(100.0f * rand() / (float)RAND_MAX);
  }
  return n;
}
TEST(ClassArrayView, Constructor) {
  std::vector<int> vec;
  int sizeVec = init1D(vec);
  int old_vec0 = vec[0];
  // Testing line 2251 of C++AMP Language and Programming Model version 1.0
  {
    Concurrency::array_view<int> av(sizeVec, vec);
    EXPECT_EQ(vec[0], av[0]);
    av[0]+=1234;
  }
  // Synchronize back at destruction time
  EXPECT_EQ(old_vec0+1234, vec[0]);
  {
    Concurrency::array_view<int> av(sizeVec, vec);
    EXPECT_EQ(vec[0], av[0]);
    old_vec0 = vec[0]++;
    av.refresh();
    EXPECT_EQ(old_vec0+1, av[0]);
  }
  // Testing line 2554 of C++AMP LPM v 1.0
  {
    int foo[]={123, 456, 789};
    Concurrency::array_view<int> av(3, foo);
    EXPECT_EQ(foo[2], av[2]);
    {
      Concurrency::array_view<int> bv(av);
      EXPECT_EQ(av[1], bv[1]);
    }
    // Line 2178 of C++AMP LPM v 1.0
    EXPECT_EQ(foo[2], av[2]);
  }
}
