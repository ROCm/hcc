// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int main()
{
    auto a_lambda_func = []() restrict(amp) {        
    };
    
    parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
       // OK. Since parallel_for_each is implemented as restrict(cpu,amp) inside
    });
   
    return 0; // Should not compile
}
