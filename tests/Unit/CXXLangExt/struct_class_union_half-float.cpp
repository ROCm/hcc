// XFAIL: *
// RUN: %hc -DTYPE="half float" %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

struct S {
  TYPE var;
};

class C {
public:
  TYPE var;
};

union U {
  TYPE var;
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

    S s;
    s.var = (TYPE)idx[0];
    C c;
    c.var = (TYPE)idx[0];
    U u;
    u.var = (TYPE)idx[0];
    p_ans[idx[0]] = (int)s.var + (int)c.var + (int)u.var;
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
