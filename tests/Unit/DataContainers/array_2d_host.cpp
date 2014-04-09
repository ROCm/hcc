// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <stdlib.h>
#include <amp.h>
#include <gtest/gtest.h>

#define N0 1000 
#define N1 1048

void init2D(std::vector<int>& vec) {
  for (int i = 0; i < N0; ++i) {
    for (int j = 0; j < N1; ++j) {
      vec.push_back(100.0f* rand() / (float)RAND_MAX);
    }
  }
}

TEST(ClassArray2D, ConstructorExtentIterator) {
  std::vector<int> v(N0*N1);
  init2D(v);
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext, v.begin());
  EXPECT_EQ(ext, arr1.extent);
  for (int i = 0; i < N0; i++)
    for (int j = 0; j < N1; j++)
      EXPECT_EQ(v[i*N1+j], arr1(i, j));
}

TEST(ClassArray2D, ConstructorSizesIterator) {
  std::vector<int> v(N0*N1);
  init2D(v);
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(N0, N1, v.begin());
  EXPECT_EQ(ext, arr1.extent);
  for (int i = 0; i < N0; i++)
    for (int j = 0; j < N1; j++)
      EXPECT_EQ(v[i*N1+j], arr1(i, j));
}

TEST(ClassArray2D, ConstructorSizes) {
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(N0, N1);
  EXPECT_EQ(ext, arr1.extent);
}

TEST(ClassArray2D, ConstructorSizesAndAccelerator) {
  Concurrency::accelerator def;
  Concurrency::accelerator_view av = def.get_default_view();
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(N0, N1, av);
  EXPECT_EQ(ext, arr1.extent);
}

TEST(ClassArray2D, ConstructorFromConstantView) {
  std::vector<int> v(N0*N1);
  init2D(v);
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext, v.begin());
  Concurrency::array_view<const int, 2>av(ext, v);
  Concurrency::array<int, 2> arr2(av);

  EXPECT_EQ(ext, arr1.extent);
  EXPECT_EQ(ext, arr2.extent);
  //Test if contents are identical
  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      EXPECT_EQ(v[i*N1+j], arr1(i, j));
      EXPECT_EQ(v[i*N1+j], arr2(i, j));
    }
  }
  arr2(15, 30) += 1;
  EXPECT_EQ(arr1(15,30)+1, arr2(15,30));
}

TEST(ClassArray2D, ConstructorFromView) {
  std::vector<int> v(N0*N1);
  init2D(v);
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext, v.begin());
  Concurrency::array_view<const int, 2>av(ext, v);
  Concurrency::array<int, 2> arr2(av);

  EXPECT_EQ(ext, arr1.extent);
  EXPECT_EQ(ext, arr2.extent);
  //Test if contents are identical
  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      EXPECT_EQ(v[i*N1+j], arr1(i, j));
      EXPECT_EQ(v[i*N1+j], arr2(i, j));
    }
  }
  arr2(15, 30) += 1;
  EXPECT_EQ(arr1(15,30)+1, arr2(15,30));
}
#if 0
TEST(ClassArray2D, CopyConstructor) {
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext);
  Concurrency::array_view<int, 2> av1(arr1);
  Concurrency::array_view<int, 2> av2(av1);
  EXPECT_EQ(ext, av2.extent);
}

TEST(ClassArray2D, Assign) {
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext);
  arr1(1, 2) = 5;
  Concurrency::extent<2> ext2(N1, N0);
  Concurrency::array<int, 2> arr2(ext2);
  arr2(1, 2) = 3;

  Concurrency::array_view<int, 2> av1(arr1);
  Concurrency::array_view<int, 2> av2(arr2);
  EXPECT_NE(ext, av2.extent);
  av2 = av1;
  EXPECT_EQ(ext, av2.extent);
  EXPECT_EQ(5, av2(1, 2));
}

TEST(ClassArray2D, Projection) {
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext);
  arr1(1, 2) = 5;
  Concurrency::array_view<int, 2> av1(arr1);
  auto av2 = av1[1];
  EXPECT_EQ(5, av2[2]);
  EXPECT_EQ(5, av1[1][2]);
}

TEST(ClassArray2D, Section) {
  Concurrency::extent<2> ext(N0, N1);
  Concurrency::array<int, 2> arr1(ext);
  arr1(1, 2) = 5;
  arr1(5, 10) = 6;
  Concurrency::array_view<int, 2> av1(arr1);
  Concurrency::array_view<int, 2> av2 =
    av1.section(1, 2, 10, 10);
  EXPECT_EQ(5, av2(0, 0)); // = av1(1, 2)
  EXPECT_EQ(6, av2(4, 8)); // = av1(5, 10)
  Concurrency::extent<2> section_ext(11,11);
  Concurrency::array_view<int, 2> av3 =
    av1.section(section_ext);
  EXPECT_EQ(6, av3(5, 10));
  Concurrency::array_view<int, 2> av4 =
    av1.section(Concurrency::index<2>(3,4));
  EXPECT_EQ(6, av4(2, 6));
  Concurrency::array_view<int, 2> av5 =
    av1.section(Concurrency::index<2>(3,4),
                section_ext);
  EXPECT_EQ(6, av4(2, 6));
}
#endif
