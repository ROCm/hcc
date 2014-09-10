//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

void foo() restrict(amp)
{
}

int f1() restrict(amp) {return 1;} 
int f2() restrict(amp) {
  return f1();
}

int CPU_Func() restrict(cpu)
{
  return f2(); // expected-error{{'f2':  no overloaded function has restriction specifiers that are compatible with the ambient context 'CPU_Func'}}
}


int main()
{
    auto a_lambda_func = []() restrict(cpu) { 
       foo(); // expected_error{{'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::<anonymous class>::operator()'}}
    };

    parallel_for_each(extent<1>(1), [](index<1>) restrict(cpu)
    {
        foo();  // expected_error{{'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::<anonymous class>::operator()'}}
    });
   
    return 1; // Should not compile
}
