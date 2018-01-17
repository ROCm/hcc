
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

  int p1 = rand() % 15 + 1;
  int p2 = rand() % 15 + 1;

  int table1[VECTOR_SIZE];
  int table2[VECTOR_SIZE];
  int table3[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table1[i] = rand() % 255 + 1;
    table2[i] = rand() % 255 + 1;
  }

  extent<1> ex(VECTOR_SIZE);
  parallel_for_each(ex, [&](index<1> idx) restrict(amp) {
    // capture multiple array types and scalar types by reference
    table3[idx[0]] = (p1 * table1[idx[0]]) + (p2 * table2[idx[0]]);
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table3[i] != (p1 * table1[i]) + (p2 * table2[i])) {
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

