// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.
#include <stdlib.h>
#include <amp.h>
#include <gtest/gtest.h>

#define N0 10
#define N1 10
#define N2 10

// The following OpenCL objects belong to C++AMP class 'accelerator'
cl_platform_id platform;
cl_device_id device;
cl_int error_code;
cl_context context;
cl_command_queue command_queue;

// This function should be part of class 'accelerator'
void clInit() {
  error_code = clGetPlatformIDs(1, &platform, NULL);
  error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
  command_queue = clCreateCommandQueue(context, device, 0, &error_code);
}

int init1D(std::vector<float>& vec) {
  int n = N0;
  for (int i = 0; i < n; ++i) {
    vec.push_back(rand() / (float)RAND_MAX);
  }
  return n;
}

TEST(ClassArray, ConstructorAndDestructor) {
  clInit();
  Concurrency::array<float> arr(N0);
}

TEST(ClassArray, Array1D) {
  std::vector<float> vec;
  int sizeVec = init1D(vec);
  clInit();
  Concurrency::array<float> arr(vec.size(), vec.begin());

  EXPECT_EQ(sizeVec, arr.get_extent().size());
  for (int i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], (arr.data())[i]);
  }
}

