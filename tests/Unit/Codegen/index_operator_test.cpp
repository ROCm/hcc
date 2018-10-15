// RUN: %cxxamp %s -o %t.out && %t.out

#include <hc.hpp>
int main(void)
{
    hc::index<1> a(1), b;
    a = b + 5566;
    return 0;
}

