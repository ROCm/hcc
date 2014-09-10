//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

int foo() restrict(amp)
{
  return 1;
}

int main()
{
  foo();    // expected-error{{'f2':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main'}}
  return 1; // Should not compile
}


