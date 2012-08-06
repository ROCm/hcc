// RUN: %gtest_amp %s -o %t1 && %t1

#include <amp.h>
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <vector>
#include <gtest/gtest.h>
using namespace Concurrency;

#define N 10

// The following OpenCL objects belong to C++AMP class 'accelerator'
cl_platform_id platform;
cl_device_id device;
cl_int error_code;
cl_context context;
cl_command_queue command_queue;

void random_init(std::vector<float>& data, int n)
{
    for (int i = 0; i < n; ++i)
	data[i] = rand() / (float)RAND_MAX;
}

class functor {
public:
  functor(array<float>& a,
          array<float>& b,
          array<float>& c)
    : a(a)
    , b(b)
    , c(c) { }

  void operator()(index<1> idx) restrict(amp) {
    c[idx] = a[idx] + b[idx];
  }

private:
  array<float>& a;
  array<float>& b;
  array<float>& c;
};

void vector_add_amp(array<float>& amp_C,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    int sz) {
}

void vector_add_cpu(//array<float>& cpu_C,
                    std::vector<float>& cpu_C,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    int sz) {
  for (int i = 0; i < sz; i++) {
    cpu_C[i] = A[i] + B[i];
  }
}

bool compare(array<float>& _exp, std::vector<float>& _ctl, int sz) {
  return false;
}

TEST(ArchitectingTest, Final)
{
  std::vector<float> A(N);
  std::vector<float> B(N);

  array<float> C_amp(N);
  std::vector<float> C_cpu(N);

  random_init(A, N);
  random_init(B, N);
  vector_add_amp(C_amp, A, B, N);
  vector_add_cpu(C_cpu, A, B, N);

  bool result = compare(C_amp, C_cpu, N);
  EXPECT_EQ(result, true); 
}

