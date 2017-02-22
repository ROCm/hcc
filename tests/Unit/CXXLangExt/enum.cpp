

// RUN: %hc -DTYPE="char"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed char"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned char"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="short"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed short"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned short"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="int"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed int"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned int"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="long"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed long"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned long"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="long long"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="signed long long"  %s -o %t.out && %t.out
// RUN: %hc -DTYPE="unsigned long long"  %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

enum E : TYPE {
  ZERO = 0
};

enum class EC : TYPE {
  ZERO = 0
};

enum struct ES : TYPE {
  ZERO = 0
};

bool test() {

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    p_ans[idx[0]] = (int)E::ZERO + (int)EC::ZERO + (int)ES::ZERO;
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs(ans[i]);
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

