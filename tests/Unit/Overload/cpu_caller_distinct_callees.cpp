// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int f(int &) restrict(amp)
{
    return 0;
}

int f(const  int &)
{
    return 1;
}

bool test()
{
    int flag = 0;
    bool passed = true;

    int v = 0;

    flag = f(v);   // in CPU path, return 1; in GPU path, return 0

    if (flag != 1)
    {
        return false;
    }

    return passed;
}

int main(int argc, char **argv)
{
    return test() ? 0 : 1;
}
