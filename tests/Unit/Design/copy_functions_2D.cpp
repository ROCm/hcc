// RUN: %gtest_amp %s -o %t && %t
#include <stdlib.h>
#include <vector>
#include <amp.h>

using namespace concurrency;

#define SIZE 1024

#include <gtest/gtest.h>

void copyBetweenArrayViewAndVector() {
  //copy from vector to array_view
  std::vector<float> vector_a(SIZE * SIZE), vector_b(SIZE * SIZE);
  for(int i = 0; i < SIZE * SIZE; ++i) {
    vector_a[i] = 0.0f;
    vector_b[i] = std::rand() / (float)RAND_MAX;
  }
  Concurrency::extent<2> e(SIZE, SIZE);
  array_view<float, 2> va(e, vector_a);
  copy(vector_b.begin(), va);
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j)
      EXPECT_EQ(va(i, j), vector_b[i * SIZE +j]);

  //copy from array_view to vector
  for(int i = 0; i < SIZE * SIZE; i++) {
    vector_a[i] = std::rand() / (float)RAND_MAX;
    vector_b[i] = 0.0f;
  }
  copy(va, vector_b.begin());
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j)
      EXPECT_EQ(vector_b[i * SIZE + j], va(i, j));
}

void copyBetweenArrayAndVector() {
  //copy data between one-dimensional array and ont-dimensional std::vector
  std::vector<float> v1(SIZE * SIZE);
  Concurrency::extent<2> e1(SIZE, SIZE);
  array<float, 2> arr1(e1);

  //copy data from one-dimensional array to ont-dimensional std::vector
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j) {
      index<2> idx(i, j);
      arr1[idx] = std::rand() / (float)RAND_MAX;
      v1[i * SIZE + j] = 0.0f;
    }
  copy(arr1, v1.begin());
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j) {
      index<2> idx(i, j);
      EXPECT_EQ(arr1[idx], v1[i * SIZE +j]);
    }

  //copy data from one-dimensional std::vector to one-dimensional array 
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j) {
      index<2> idx(i, j);
      arr1[idx] = 0.0f;
      v1[i] = std::rand() / (float)RAND_MAX;
    }
  copy(v1.begin(), arr1);
  for(int i = 0; i < SIZE; ++i)
    for(int j = 0; j < SIZE; ++j) {
      index<2> idx(i, j);
      EXPECT_EQ(arr1[idx], v1[i * SIZE + j]);
    }
}

TEST(Copy, ArrayViewAndVector2D) {
  copyBetweenArrayViewAndVector();
}

TEST(Copy, ArrayAndVector2D) {
  copyBetweenArrayAndVector();
}
