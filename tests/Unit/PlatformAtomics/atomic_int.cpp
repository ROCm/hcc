
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <random>
#include <atomic>
#include <amp.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with platform atomics functions
// requires HSA Full Profile to operate successfully

bool test() {

  // define inputs and output
  const int vecSize = 2048;

  std::atomic_int table_a[vecSize];
  std::atomic_int table_b[vecSize];
  std::atomic_int table_c[vecSize];
  auto ptr_a = &table_a[0];
  auto ptr_b = &table_b[0];
  auto ptr_c = &table_c[0];

  // initialize test data
  std::random_device rd;
  std::uniform_int_distribution<int32_t> int_dist;
  for (int i = 0; i < vecSize; ++i) {
    table_a[i].store(int_dist(rd));
    table_b[i].store(int_dist(rd));
  }

  // launch kernel
  Concurrency::extent<1> e(vecSize);
  parallel_for_each(
    e,
    [=](Concurrency::index<1> idx) restrict(amp) {

      int tid = idx[0];
      (ptr_a + tid)->fetch_add(1);
      (ptr_b + tid)->fetch_sub(1);
      (ptr_c + tid)->store(0);
      (ptr_c + tid)->fetch_add((ptr_a + tid)->load(std::memory_order_acquire), std::memory_order_release);
      (ptr_c + tid)->fetch_add((ptr_b + tid)->load(std::memory_order_seq_cst), std::memory_order_acq_rel);

  });

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
