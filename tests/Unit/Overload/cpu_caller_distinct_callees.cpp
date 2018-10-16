// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;


int f(int &) [[hc]]
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
