
// RUN: %hc %s -o %t.out && %t.out

#include <hc/hc.hpp>

#include <iostream>

// added for checking HSA profile
#include <hc/hc.hpp>

// test C++AMP with fine-grained SVM
// requires HSA Full Profile to operate successfully

#define SIZE (16)

using namespace hc;

// test supply a class with operator() to parallel_for_each
// the class will call a separate functor
class user_functor {
public:
  void operator()(index<1>& idx, int (&input)[SIZE]) const [[hc]] {
    input[idx[0]] = idx[0];
  }
};

class prog {
  int (&input)[SIZE];
  user_functor& kernel;

public:
  prog(int (&t)[SIZE], user_functor& f) [[cpu, hc]] : input(t), kernel(f) {
  }

  void operator()(index<1>& idx) const [[hc]] {
    kernel(idx, input);
  }

  void run() {
    parallel_for_each(extent<1>(SIZE), *this);
  }

  // verify output
  bool test() {
    bool ret = true;
    for (int i = 0; i < SIZE; ++i) {
      if (input[i] != i) {
        ret = false;
        break;
      }
    }
    return true;
  }
};

int main() {
  bool ret = true;
 
  // only conduct the test in case we are running on a HSA full profile stack
  hc::accelerator acc;
  if (acc.is_hsa_accelerator() &&
      acc.get_profile() == hc::accelerator_profile_full) {

    // prepare test data
    int input[SIZE] { 0 };
  
    // launch kernel
    user_functor kernel;
    prog p(input, kernel);
    p.run();
  
    // check result
    ret &= p.test();

  }

  return !(ret == true);
}

