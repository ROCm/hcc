// RUN: %gtest_amp %s -o %t && %t
#include <stdlib.h>
#include <vector>
#include <amp.h>

using namespace concurrency;

#define SIZE 4

#include <gtest/gtest.h>

void copyBetweenArrayViewAndVector() {
  //copy from vector to array_view
  std::vector<float> vector_a(SIZE), vector_b(SIZE);
  for (int i = 0; i < SIZE; ++i) {
    vector_a[i] = 0.0f;
    vector_b[i] = std::rand() / (float)RAND_MAX;
  }
  Concurrency::extent<1> e(SIZE);
  array_view<float, 1> va(e, vector_a);
  copy(vector_b.begin(), va);
  for (int i = 0; i < SIZE; ++i)
    EXPECT_EQ(va[i], vector_b[i]);

  //copy from array_view to vector
  for (int i = 0; i < SIZE; i++) {
    vector_a[i] = std::rand() / (float)RAND_MAX;
    vector_b[i] = 0.0f;
  }
  copy(va, vector_b.begin());
  for (int i = 0; i < SIZE; ++i) {
    EXPECT_EQ(vector_b[i], va[i]);
  }
}

void copyBetweenArrayAndVector() {
  //copy data between one-dimensional array and ont-dimensional std::vector
  std::vector<float> v1(SIZE);
  Concurrency::extent<1> e1(SIZE);
  array<float, 1> arr1(e1);
  
  //copy data from one-dimensional array to ont-dimensional std::vector
  for (int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    arr1[idx] = std::rand() / (float)RAND_MAX;
    v1[i] = 0.0f;
  }

  copy(arr1, v1.begin());

  for (int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    EXPECT_EQ(arr1[idx], v1[i]);
  }  
  
  //copy data from one-dimensional std::vector to one-dimensional array 
  for (int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    arr1[idx] = 0.0f;
    v1[i] = std::rand() / (float)RAND_MAX;
  }
  copy(v1.begin(), arr1);
  for (int i = 0; i < SIZE; ++i) {
    index<1> idx(i);
    EXPECT_EQ(arr1[idx], v1[i]);
  }  
}

TEST(Copy, ArrayViewAndVector) {
  copyBetweenArrayViewAndVector();
}

TEST(Copy, ArrayAndVector) {
  copyBetweenArrayAndVector();
}
