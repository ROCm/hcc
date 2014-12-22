// XFAIL: Linux

// RUN: %cxxamp -DTYPE="char" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="signed char" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="unsigned char" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="short" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="signed short" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="unsigned short" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="int" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="signed int" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="unsigned int" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="long" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="signed long" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="unsigned long" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="long long" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="signed long long" -Xclang -fhsa-ext %s -o %t.out && %t.out
// RUN: %cxxamp -DTYPE="unsigned long long" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="float" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="double" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="long double" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="bool" -Xclang -fhsa-ext %s -o %t.out && %t.out

// RUN: %cxxamp -DTYPE="wchar_t" -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

TYPE func(TYPE arg) restrict(amp)
{
  TYPE local = arg;
  return local;
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

    p_ans[idx[0]] = func((TYPE)idx[0]);
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs((TYPE)ans[i] - (TYPE)i);
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}
