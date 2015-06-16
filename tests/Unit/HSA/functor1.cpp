// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

#define SIZE (16)

using namespace concurrency;

// test supply a class with operator() to parallel_for_each
class prog {
  int (&input)[SIZE];

public:
  prog(int (&t)[SIZE]) : input(t) {
    run();
  }

  void operator() (index<1>& idx) restrict(amp) {
    input[idx[0]] = idx[0];
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
  prog p(input);

  // check result
  ret &= p.test();

  return !(ret == true);
}

