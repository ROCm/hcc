//XFAIL:*
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

int f1() restrict(cpu) {return 1;} 
int f2() restrict(amp) {return 2;}

int AMP_AND_CPU_Func() restrict(cpu,amp)
{
  return f2(); // expected-error{{undefined reference to `f2()'}}  // Since in CPU path, there is no any cpu restricted 'f2'
}

int AMP_AND_CPU_Func_1() restrict(cpu,amp)
{
  return f1(); // expected-error{{call from AMP-restricted function to CPU-restricted function}}
}

int foo() {}

int main()
{
    auto a_lambda_func = []() restrict(cpu,amp) { 
       foo(); // lambda:expected_error{{'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::<anonymous class>::operator()'}}
    };

    parallel_for_each(extent<1>(1), [](index<1>) restrict(cpu,amp)
    {
        foo();  // pfe:expected_error{{'foo':  no overloaded function has restriction specifiers that are compatible with the ambient context 'main()::<anonymous class>::operator()'}}
    });
   
    return 1; // Should not compile
}
