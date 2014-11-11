// XFAIL: *
// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

void foo()
{
}

int main()
{
    parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
        foo();  // Call from AMP to CPU. Caller: Lambda
    });
}
