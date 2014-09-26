// XFAIL: *

// RUN: %amp_device -DTYPE="half float" -D__GPU__ -Xclang -fhsa-ext %s -m64 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp -DTYPE="half float" -Xclang -fhsa-ext %link %t/kernel.o %s -o %t.out && %t.out

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
