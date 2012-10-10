// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <stdlib.h>
#include <amp.h>
#include <gtest/gtest.h>

#define N0 48 
#define N1 52 

void init1D(std::vector<int>& vec) {
  for (int i = 0; i < N0; ++i) {
    vec.push_back(100.0f* rand() / (float)RAND_MAX);
  }
}


TEST(ClassArrayView1D, Assign) {
  Concurrency::extent<1> ext(N0);
  Concurrency::array<int, 1> arr1(ext);
  arr1[12] = 5;
  Concurrency::extent<1> ext2(N1);
  Concurrency::array<int, 1> arr2(ext2);
  arr2[12] = 3;

  Concurrency::array_view<int, 1> av1(arr1);
  Concurrency::array_view<int, 1> av2(arr2);
  EXPECT_NE(ext, av2.extent);
  av2 = av1;
  EXPECT_EQ(ext, av2.extent);
  EXPECT_EQ(5, av2[12]);
}

TEST(ClassArrayView1D, Section) {
  Concurrency::extent<1> ext(N0);
  Concurrency::array<int, 1> arr1(ext);
  arr1[12] = 5;
  Concurrency::array_view<int, 1> av1(arr1);
  Concurrency::array_view<int, 1> av2 =
    av1.section(3, 10);
  EXPECT_EQ(5, av2[9]); // = av1[12]
  Concurrency::extent<1> section_ext(11);
  Concurrency::array_view<int, 1> av3 =
    av1.section(section_ext);
  EXPECT_EQ(5, av3[12]);

  Concurrency::extent<1> ext3 = av3.extent;
  EXPECT_EQ(11, ext3[0]);


  Concurrency::array_view<int, 1> av4 =
    av1.section(Concurrency::index<1>(10));
  Concurrency::extent<1> ext4 = av4.extent;
  EXPECT_EQ(N0-10, ext4[0]);

  Concurrency::array_view<int, 1> av5 =
    av1.section(Concurrency::index<1>(10),
      section_ext);
  EXPECT_EQ(5, av5[2]);

  Concurrency::extent<2> ext6(4, N0/4);
  Concurrency::array_view<int, 2> av6 =
    av1.view_as<2>(ext6);
  EXPECT_EQ(5, av6(1, 0));

  int *p = av5.data();
  EXPECT_EQ(5, p[2]);
}
