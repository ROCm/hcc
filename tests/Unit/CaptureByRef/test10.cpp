
// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <iostream>
#include <cstdlib>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

#define VECTOR_SIZE (1024)

struct POD {
  int foo;
  int bar;
};

bool test() {
  using namespace hc;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  POD p;
  p.foo = rand() % 15 + 1;
  p.bar = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  array_view<int, 1> av(ex, table);
  parallel_for_each(av.get_extent(), [&, av](hc::index<1> idx) [[hc]] {
    // capture POD type by reference
    av[idx] *= (p.foo + p.bar);
  });

  av.synchronize();

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != i * (p.foo + p.bar)) {
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

