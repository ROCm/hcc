// RUN: %gtest_amp %s -o %t && %t
#include <stdlib.h>
#include <vector>
#include <amp.h>

using namespace concurrency;

#define SIZE 1024

#include <gtest/gtest.h>

void copyBetweenArray() {
  //copy data between one-dimensional array_view and one-dimensional array
  array<float, 1> arr1(SIZE), arr2(SIZE);
  for(int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    arr1[idx] = rand() / (float)RAND_MAX;
    arr2[idx] = 0.0f;
  }
  copy(arr1, arr2);
  for(int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    EXPECT_EQ(arr2[idx], arr1[idx]);
  }

  //copy data between two-dimensional array_view and two-dimensional array
  Concurrency::extent<2> e(SIZE, SIZE);
  array<float, 2> arr3(e), arr4(e);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      index<2> idx(i, j);
      arr3[idx] = rand() / (float)RAND_MAX;
      arr4[idx] = 0.0f;
    }
  }
  copy(arr3, arr4);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      EXPECT_EQ(arr3(i, j), arr4(i, j));
    }
  }
}

void copyBetweenArrayView() {
  //copy data between one-dimensional array_view and one-dimensional array
  array<float, 1> arr1(SIZE), arr2(SIZE);
  array_view<float, 1> av1(arr1),av2(arr2);
  for(int i = 0; i < SIZE; ++i) {
    arr1(i) = 0.0f;
    arr2(i) = rand() / (float)RAND_MAX;
  }
  copy(av1, av2);
  for(int i = 0; i < SIZE; ++i) {
    EXPECT_EQ(av1(i), av2(i));
  }

  //copy data between two-dimensional array_view and two-dimensional array
  Concurrency::extent<2> e(SIZE, SIZE);
  array<float, 2> arr3(e), arr4(e);
  array_view<float, 2> av3(arr3), av4(arr4);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      arr3(i, j) = rand() / (float)RAND_MAX;
      arr4(i, j) = 0.0f;
    }
  }
  copy(arr3, arr4);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      EXPECT_EQ(arr3(i, j), arr4(i, j));
    }
  }
}

void copyBetweenArrayAndArrayView() {
  //copy data between one-dimensional array_view and one-dimensional array
  array<float, 1> arr1(SIZE), arr2(SIZE), arr3(SIZE);
  array_view<float, 1> av(arr1);
  for(int i = 0; i < SIZE; ++i) {
    arr1(i) = 0.0f;
    arr2(i) = rand() / (float)RAND_MAX;
    arr3(i) = 0.0f;
  }
  copy(arr2, av);
  copy(av, arr3);
  for(int i = 0; i < SIZE; ++i) {
    EXPECT_EQ(arr2(i), arr3(i));
  }

  //copy data between two-dimensional array_view and two-dimensional array
  Concurrency::extent<2> e(SIZE, SIZE);
  array<float, 2> arr4(e), arr5(e), arr6(e);
  array_view<float, 2> av2(arr4);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      arr4(i, j) = 0.0f;
      arr5(i, j) = rand() / (float)RAND_MAX;
      arr6(i, j) =0.0f;
    }
  }
  copy(arr5, av2);
  copy(av2, arr6);
  for(int i = 0; i < SIZE; ++i) {
    for(int j = 0; j < SIZE; ++j) {
      EXPECT_EQ(arr5(i, j), arr6(i, j));
    }
  }
}

TEST(Copy, ArrayViewAndArray) {
  copyBetweenArrayAndArrayView();
}

TEST(Copy, Array) {
  copyBetweenArray();
}

TEST(Copy, ArrayView) {
  copyBetweenArrayView();
}


