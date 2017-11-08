// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;


int main()
{
    parallel_for_each(extent<1>(1), [](index<1>) restrict(amp)
    {
       // OK. Since parallel_for_each is implemented as restrict(cpu,amp) inside
    });

    auto a_lambda_func = []() restrict(amp) {
    };

    return 0; // Should not compile
}
