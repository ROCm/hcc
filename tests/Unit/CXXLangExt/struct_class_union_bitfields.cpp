// XFAIL: Linux

// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out

#include <amp.h>

struct S {
  unsigned int bit : 3;
};

class C {
public:
  unsigned int bit : 3;
};

union U {
  unsigned int bit : 3;
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
    s.bit = 7;
    ++s.bit;
    C c;
    c.bit = 7;
    ++c.bit;
    U u;
    u.bit = 7;
    ++u.bit;
    p_ans[idx[0]] = (int)s.bit + (int)c.bit + (int)u.bit;
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
  return (error != 0);
}
