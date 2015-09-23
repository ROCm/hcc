// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

#define SIZE (16)

using namespace concurrency;

// test supply a class with operator() to parallel_for_each
// the class will call a separate functor
class user_functor {
public:
  void operator() (index<1>& idx, int (&input)[SIZE]) restrict(amp) {
    input[idx[0]] = idx[0];
  }
};

class prog {
  int (&input)[SIZE];
  user_functor& kernel;

public:
  prog(int (&t)[SIZE], user_functor& f) : input(t), kernel(f) {
  }

  void operator() (index<1>& idx) restrict(amp) {
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
 
  // prepare test data
  int input[SIZE] { 0 };

  // launch kernel
  user_functor kernel;
  prog p(input, kernel);
  p.run();

  // check result
  ret &= p.test();

  return !(ret == true);
}

