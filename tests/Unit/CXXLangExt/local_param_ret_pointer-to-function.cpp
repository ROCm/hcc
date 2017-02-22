
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

// added for checking HSA profile
#include <hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

int func(float arg1, char arg2, char arg3) restrict(amp, cpu) 
{
  return (int)(arg2 + arg3);
}

bool test() {

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    int (*pt2Function)(float, char, char) = &func;
    p_ans[idx[0]] = (*pt2Function)(0, (char)idx[0], (char)idx[0]);
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs(ans[i] - func(0.0, (char)i, (char)i));
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

