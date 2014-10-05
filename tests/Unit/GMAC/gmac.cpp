// XFAIL: hsa
// RUN: %gtest_amp %s -o %t1 && %t1

#include <gtest/gtest.h>
#include <gmac/opencl.h>

const char *kernel_source = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    c[i] = a[i] + b[i];\
}\
";

TEST(GMAC, vecAdd) {
  const int vecSize = 16;
  ecl_error error_code;
  ecl_kernel kernel;
  float *a, *b, *c;
  
  error_code = eclCompileSource(kernel_source);
  ASSERT_EQ(eclSuccess, error_code);
  error_code = eclGetKernel("vecAdd", &kernel);
  ASSERT_EQ(eclSuccess, error_code);

  // Alloc & init input data
  error_code = eclMalloc((void **)&a, vecSize * sizeof(float));
  ASSERT_EQ(eclSuccess, error_code);
  error_code = eclMalloc((void **)&b, vecSize * sizeof(float));
  ASSERT_EQ(eclSuccess, error_code);
  // Alloc output data
  error_code = eclMalloc((void **)&c, vecSize * sizeof(float));
  ASSERT_EQ(eclSuccess, error_code);

  float sum = 0.f;

  for(unsigned i = 0; i < vecSize; i++) {
          a[i] = 1.f * rand() / RAND_MAX;
          b[i] = 1.f * rand() / RAND_MAX;
          sum += a[i] + b[i];
  }
  // Call the kernel    
  size_t global_size = vecSize;

  error_code = eclSetKernelArgPtr(kernel, 0, c);
  ASSERT_EQ(eclSuccess, error_code);
  error_code = eclSetKernelArgPtr(kernel, 1, a);
  ASSERT_EQ(eclSuccess, error_code);
  error_code = eclSetKernelArgPtr(kernel, 2, b);
  ASSERT_EQ(eclSuccess, error_code);
  error_code = eclSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
  ASSERT_EQ(eclSuccess, error_code);

  error_code = eclCallNDRange(kernel, 1, NULL, &global_size, NULL);
  ASSERT_EQ(eclSuccess, error_code);

  float error = 0.f;
  float check = 0.f;
  for(unsigned i = 0; i < vecSize; i++) {
    error += c[i] - (a[i] + b[i]);
    check += c[i];
  }

  EXPECT_EQ(sum, check);

  /* Clean up resources */
  eclFree(a);
  eclFree(b);
  eclFree(c);
}

