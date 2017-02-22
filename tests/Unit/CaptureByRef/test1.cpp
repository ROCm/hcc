
// RUN: %hc %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>
#include <cstdlib>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

#define VECTOR_SIZE (1024)

bool test() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  int val = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  array_view<int, 1> av(ex, table);
  parallel_for_each(av.get_extent(), [&, av](index<1> idx) restrict(amp) {
    // capture scalar type by reference
    av[idx] *= (val * val);
  });

  av.synchronize();

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != i * (val * val)) {
      std::cout << "Failed at " << i << std::endl;
      return false;
    }
  }

  std::cout << "Passed" << std::endl;
  return true;
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

