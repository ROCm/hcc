// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <random>
#include <future>
#include <amp.h>

// An HSA version of C++AMP program
int main ()
{
  // define inputs and output
  const int vecSize = 4096;
  const int dimSize = 16;

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

  // launch kernel
  Concurrency::extent<3> e(dimSize, dimSize, dimSize);
  Concurrency::completion_future fut = Concurrency::async_parallel_for_each(
    e.tile<2, 2, 2>(),
    [=](Concurrency::tiled_index<2, 2, 2> idx) restrict(amp) {
      int fidx = idx.global[0] * dimSize * dimSize + idx.global[1] * dimSize + idx.global[2];
      for (int i = 0; i < 1024; ++i) 
        p_c[fidx] = p_a[fidx] + p_b[fidx];

  });

  fut.wait();

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

