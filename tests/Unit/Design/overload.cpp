// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;

int f() [[hc]] { return 55; }
int f() [[cpu]] { return 66; }
int g() [[cpu, hc]] { return f(); }

bool TestOnHost()
{
    return g() == 66;
}

bool TestOnDevice()
{
    array<int, 1> a((extent<1>(1)));
    array_view<int> A(a);
    extent<1> ex(1);
    parallel_for_each(ex, [&](hc::index<1> idx) [[cpu, hc]] {
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
