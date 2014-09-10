// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>

class c2
{
public:
    int f(int) restrict(cpu)
    {
        return 1;
    }

    int f(float) restrict(cpu,amp)
    {
        return 0;
    }

};

class c1
{
public:
    int b(int) restrict(auto) // Use 'auto' to select good compilation path
    {
        c2 o;  // Check SMF is after the 'auto' inferring

        int i;

        return o.f(i); // if not inferred,  undefined reference to `c2::f(int)'
    }
};

bool test()
{
    c1 o;

    int i = 0;

    int flag = o.b(i);

    return ((flag == 1) ? true : false);
}

int main(int argc, char **argv)
{
    int ret = test();

    return ret?0:1;
}


