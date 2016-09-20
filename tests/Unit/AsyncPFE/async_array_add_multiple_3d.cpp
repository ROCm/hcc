
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <random>
#include <future>
#include <vector>
#include <algorithm>
#include <utility>
#include <hc.hpp>

// FIXME: HSA runtime seems buggy in case LOOP_COUNT is very big
// (ex: 1024 * 1024).
#define LOOP_COUNT (1)

// test HC with fine-grained SVM
// requires HSA Full Profile to operate successfully
// An example which shows how to launch a kernel asynchronously
bool test() {
  // define inputs and output
  const int vecSize = 2048;
  const int dimSize = 8;

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

  // the vector to store handles to each async pfe 
  std::vector<hc::completion_future> futures;

  // divide the array into 4 quarters
  // each quarter contains 512 elements
  // treat each quarter as a 8*8*8 3D array
  hc::extent<3> e(dimSize, dimSize, dimSize);

#define ASYNC_KERNEL_DISPATCH(x, y) \
  hc::parallel_for_each( \
    e, \
    [=](hc::index<3> idx) restrict(amp) { \
      const int offset = vecSize / (x) * (y); \
      const int fidx = idx[0] * dimSize * dimSize + idx[1] * dimSize + idx[2]; \
      for (int i = 0; i < LOOP_COUNT; ++i) \
        p_c[fidx + offset] = p_a[fidx + offset] + p_b[fidx + offset]; \
  })

  // asynchronously launch each quarter
  futures.push_back(std::move(ASYNC_KERNEL_DISPATCH(4, 0)));
  futures.push_back(std::move(ASYNC_KERNEL_DISPATCH(4, 1)));
  futures.push_back(std::move(ASYNC_KERNEL_DISPATCH(4, 2)));
  futures.push_back(std::move(ASYNC_KERNEL_DISPATCH(4, 3)));

  // wait for all kernels to finish execution
  std::for_each(futures.cbegin(), futures.cend(), [](const hc::completion_future& fut) { fut.wait(); });

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

  return (error == 0);
}

int main() {
  bool ret = true;

  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::hcAgentProfileFull) {
    ret &= test();
  }

  return !(ret == true);
}

