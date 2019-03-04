// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;


int main()
{
    // This test outlines a subtle issue with how we obtain mangled kernel names
    // which is tracked in SWDEV-137849. a_lambda_func is moved after the pfe to
    // work around this and ensure matched mangling.
    parallel_for_each(extent<1>(1), [](hc::index<1>) [[hc]]
    {
       // OK. Since parallel_for_each is implemented as [[cpu, hc]] inside
    });

    auto a_lambda_func = []() [[hc]] {
    };

    return 0; // Should not compile
}
