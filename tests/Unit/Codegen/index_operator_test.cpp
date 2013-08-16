// RUN: %cxxamp %s %link

#include <amp.h>
int main(void)
{
    concurrency::index<1> a(1), b;
    a = b + 5566;
    return 0;
}

