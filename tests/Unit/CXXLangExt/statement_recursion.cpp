// XFAIL: *
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

int fib(int x) restrict (amp, cpu) {
  if (x == 0) return 0;
  if (x == 1) return 1;
  return fib(x - 1) + fib(x - 2);
}

// An HSA version of C++AMP program
int main ()
{

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    p_ans[idx[0]] = fib(idx[0]);
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs(ans[i] - fib(i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}

