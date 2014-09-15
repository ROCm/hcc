// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <random>
#include <amp.h>

// An HSA version of C++AMP program
int main ()
{
  // define inputs and output
  const int vecSize = 2048;

  int table_a[vecSize];
  int table_b[vecSize];
  int table_c[vecSize];
  int *p_a = &table_a[0];
  int *p_b = &table_b[0];
  int *p_c = &table_c[0];

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i] = int_dist(rd);
    table_b[i] = int_dist(rd);
  }

  // divide the array into 4 quarters
  Concurrency::extent<1> e(vecSize / 4);

  // the first quarter of the array
  void *handle1 = Concurrency::async_parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {
      p_c[idx[0]] = p_a[idx[0]] + p_b[idx[0]];
  });

  // the second quarter of the array
  void *handle2 = Concurrency::async_parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {
      p_c[idx[0] + vecSize/4] = p_a[idx[0] + vecSize/4] + p_b[idx[0] + vecSize/4];
  });

  // the third quarter of the array
  void *handle3 = Concurrency::async_parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {
      p_c[idx[0] + (vecSize/4) * 2] = p_a[idx[0] + (vecSize/4) * 2] + p_b[idx[0] + (vecSize/4) * 2];
  });

  // the foruth quarter of the array
  void *handle4 = Concurrency::async_parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {
      p_c[idx[0] + (vecSize/4) * 3] = p_a[idx[0] + (vecSize/4) * 3] + p_b[idx[0] + (vecSize/4) * 3];
  });


  Concurrency::async_wait_kernel_complete(handle1);
  Concurrency::async_wait_kernel_complete(handle2);
  Concurrency::async_wait_kernel_complete(handle3);
  Concurrency::async_wait_kernel_complete(handle4);

  // verify
  int error = 0;
  for(unsigned i = 0; i < vecSize; i++) {
    error += table_c[i] - (table_a[i] + table_b[i]);
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return error != 0;
}

