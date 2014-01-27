// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O3 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include <amp.h>
using namespace Concurrency;

int f() restrict(amp) { return 55; }
int f() restrict(cpu) { return 66; }
int g() restrict(amp,cpu) { return f(); }

bool TestOnHost()
{
    return g() == 66;
}

bool TestOnDevice()
{
    array<int, 1> A((extent<1>(1)));
    extent<1> ex(1);
    parallel_for_each(ex, [&](index<1> idx) restrict(amp,cpu) {
        A(idx) = g();
    });
    return A[0] == 55;
}

int main()
{
    int result = 1;
    result &= TestOnHost();
    result &= TestOnDevice();
    return !result;
}
