// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <stdlib.h>
#include <amp.h>
#include <gtest/gtest.h>

#define N0 5000

int init1D(std::vector<float>& vec) {
  int n = N0;
  for (int i = 0; i < n; ++i) {
    vec.push_back(rand() / (float)RAND_MAX);
  }
  return n;
}
TEST(ClassArray, ConstructorAndDestructor) {
  Concurrency::array<float> arr1(N0);
  std::vector<float> vec;
  int sizeVec = init1D(vec);
  Concurrency::array<float> arr(vec.size(), vec.begin());
  Concurrency::array_view<float> av(arr);
  Concurrency::array_view<float> av2(av);

  EXPECT_EQ(sizeVec, arr.get_extent().size());
  for (int i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], (arr.data())[i]);
    EXPECT_EQ(vec[i], arr[Concurrency::index<1>(i)]);
    EXPECT_EQ(vec[i], av[i]);
    EXPECT_EQ(vec[i], av2[i]);
  }
  Concurrency::extent<1> e(vec.size());
  Concurrency::array<int> arr2(e);
  Concurrency::accelerator def;
  Concurrency::array<int>(N0, def.get_default_view());

  {
    Concurrency::extent<1> ext1d(vec.size());
    Concurrency::array<float> arr_ext(ext1d, vec.begin());
  }

  {
    Concurrency::extent<1> ext1d(vec.size());
    Concurrency::array<float> arr_ext(ext1d, vec.begin(),
      vec.end());
    for (int i = 0; i < vec.size(); ++i) {
      EXPECT_EQ(vec[i], arr_ext[i]);
    }
  }
  {
    Concurrency::array<float> arr_ext(vec.size(),
      vec.begin(), vec.end());
    for (int i = 0; i < vec.size(); ++i) {
      EXPECT_EQ(vec[i],
        arr_ext(Concurrency::index<1>(i)));
    }
    Concurrency::extent<2> ext(4, N0/4);
    Concurrency::array_view<float, 2> av =
      arr_ext.view_as<2>(ext);
    EXPECT_EQ(vec[N0/4], av(1, 0));
  }

  {
    const Concurrency::array<float> arr_c(vec.size(),
      vec.begin());
    EXPECT_EQ(vec[10], arr_c(10));
  }
  {
    const Concurrency::array<float> arr_c(arr);
    EXPECT_EQ(vec[10], arr_c(10));
  }
}
