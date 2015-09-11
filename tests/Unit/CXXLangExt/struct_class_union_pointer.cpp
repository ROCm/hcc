// XFAIL: Linux

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

// RUN: %hc -DTYPE="float"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="double"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="long double"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="bool"  %s -o %t.out && %t.out

// RUN: %hc -DTYPE="wchar_t"  %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

struct S {
  TYPE * ptr;
};

class C {
public:
  TYPE * ptr;
};

union U {
  TYPE * ptr;
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
    S s;
    s.ptr = &var;
    C c;
    c.ptr = &var;
    U u;
    u.ptr = &var;
    p_ans[idx[0]] = (int)(*s.ptr) + (int)(*c.ptr) + (int)(*u.ptr);
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    error += abs((TYPE)ans[i] - (TYPE)(3 * i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}
