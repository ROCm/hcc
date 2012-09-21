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
    , c(c) {
      CreateCLProgram();
      CreateCLKernel();
    }

  // Compiler generates this function to create the cl program
  void CreateCLProgram();
  void CreateCLKernel();

  // Compiler locates this function
  void operator()(index<1> idx) restrict(amp) {
    c[idx] = a[idx] + b[idx];
  }

  // Compiler generates a function to setup the kernel arguments
  void setArgs() const {
    cl_mem a_device = clGetBuffer(context, a.data());
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_device);
    assert(error_code == CL_SUCCESS);
    cl_mem b_device = clGetBuffer(context, b.data());
    error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_device);
    assert(error_code == CL_SUCCESS);
    cl_mem c_device = clGetBuffer(context, c.data());
    error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_device);
    assert(error_code == CL_SUCCESS);
  }

  void launchOpenCL(size_t* gws, int dim) const {
    error_code = clEnqueueNDRangeKernel(command_queue, kernel, dim, NULL, gws, NULL, 0, NULL, NULL);
    assert(error_code == CL_SUCCESS);

    error_code = clFinish(command_queue);
    assert(error_code == CL_SUCCESS);
  }

  // Compiler generates the OpenCL kernel function 
  const char* kernel_source ="\
                              __kernel void fn(__global float* a, __global float* b, __global float* c) {\
                                int x = get_global_id(0);\
                                \
                                int idx = x;\
                                \
                                c[idx] = a[idx] + b[idx];\
                              }\
                             ";

  // The following OpenCL objects belong to C++AMP class 'accelerator'
  cl_program program;
  cl_kernel kernel;

private:
  array<float>& a;
  array<float>& b;
  array<float>& c;
};

// This function should be part of class 'accelerator'
void clInit() {
  device = Concurrency::accelerator().clamp_get_device_id();
  context = Concurrency::accelerator().get_default_view().clamp_get_context(); 
  assert(error_code == CL_SUCCESS);
  command_queue = Concurrency::accelerator().get_default_view().clamp_get_command_queue();
  assert(error_code == CL_SUCCESS);
}

void functor::CreateCLProgram() {
  program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
  assert(error_code == CL_SUCCESS);
  error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  assert(error_code == CL_SUCCESS);
}

void functor::CreateCLKernel() {
  kernel = clCreateKernel(program, "fn", &error_code);
  assert(error_code == CL_SUCCESS);
}

void vector_add_amp(array<float>& amp_C,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    int sz) {
  array<float> amp_A(A.size(), A.begin());
  array<float> amp_B(B.size(), B.begin());
  extent<1> e(sz);
  parallel_for_each(e, functor(amp_A, amp_B, amp_C));
}

void vector_add_cpu(//array<float>& cpu_C,
                    std::vector<float>& cpu_C,
                    const std::vector<float>& A,
                    const std::vector<float>& B,
                    int sz) {
  for (int i = 0; i < sz; i++) {
    //cpu_C[i] = A[i] + B[i];
    cpu_C[i] = A[i] + B[i];
  }
}

bool compare(array<float>& _exp, std::vector<float>& _ctl, int sz) {
  for (int i = 0; i < sz; i++) {
    //if (_exp[i] != _ctl[i]) return false;
    std::cout << "gpu: " << (_exp.data())[i] << ", cpu: " << _ctl[i] << "\n";
    if ((_exp.data())[i] != _ctl[i]) return false;
  }
  return true;
}

TEST(ArchitectingTest, Final)
{
  clInit();
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

