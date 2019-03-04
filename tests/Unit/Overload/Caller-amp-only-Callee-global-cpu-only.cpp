// XFAIL: *
// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;

void foo()
{
}

int main()
{
    parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
    {
        foo();  // Call from AMP to CPU. Caller: Lambda
    });
}
