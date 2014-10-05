// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>

class B
{
public:
    float f1(int &flag)
    {
        flag = 1;
        return 0.0;
    }
};

bool test() restrict(auto)
{
    bool passed = true;
    int flag = 0;

    class D: public B
    {
    public:
        float f2(int &flag) {return 0.0;}
    };

    D o;

    o.f1(flag); // OK since test is inferred as CPU

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


