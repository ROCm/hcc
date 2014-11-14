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

struct S {
  TYPE & ref;
  explicit S(TYPE &var) restrict (amp) : ref(var) {};
};

class C {
public:
  TYPE & ref;
  explicit C(TYPE &var) restrict (amp) : ref(var) {};
};

// An HSA version of C++AMP program
int main ()
{

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    TYPE var = (TYPE)idx[0];
    S s(var);
    C c(var);
    p_ans[idx[0]] = (int)(s.ref) + (int)(c.ref);
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs((TYPE)ans[i] - (TYPE)(2 * i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}
