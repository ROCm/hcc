// RUN: %cxxamp %s -o %t.out && %t.out
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
    array<int, 1> a((extent<1>(1)));
    array_view<int> A(a);
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
