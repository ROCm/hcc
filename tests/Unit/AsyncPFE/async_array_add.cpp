// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <iostream>
#include <random>
#include <future>

// FIXME: remove C++AMP dependency
#include <amp.h>
#include <hc.hpp>

// FIXME: HSA runtime seems buggy in case LOOP_COUNT is very big
// (ex: 1024 * 1024).
#define LOOP_COUNT (1)

// An example which shows how to launch a kernel asynchronously
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

  // launch kernel
  hc::extent<1> e(vecSize);
  hc::completion_future fut = hc::async_parallel_for_each(
    e,
    [=](hc::index<1> idx) restrict(amp) {
      for (int i = 0; i < LOOP_COUNT; ++i) 
        p_c[idx[0]] = p_a[idx[0]] + p_b[idx[0]];

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

