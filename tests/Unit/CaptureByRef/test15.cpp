
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>
#include <iostream>
#include <cstdlib>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

#define VECTOR_SIZE (1024)

class POD {
public:
  int foo;
  int bar;
};

class POD2 : public POD {
public:
  int baz;
};

class POD3 : public POD2 {
public:
  int qux;
};

bool test() {
  using namespace Concurrency;

  int table[VECTOR_SIZE];
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    table[i] = i;
  }

  POD3 p;
  p.foo = rand() % 15 + 1;
  p.bar = rand() % 15 + 1;
  p.baz = rand() % 15 + 1;
  p.qux = rand() % 15 + 1;

  extent<1> ex(VECTOR_SIZE);
  parallel_for_each(ex, [&](index<1> idx) restrict(amp) {
    // capture array type, and an inherited type by reference
    table[idx[0]] = (p.foo * p.bar * p.baz * p.qux);
  });

  // verify result
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    if (table[i] != (p.foo * p.bar * p.baz * p.qux)) {
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

