
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <hc.hpp>

[[hc]] int flag;

void foo(bool set) [[cpu, hc]]
{
    flag = set ? 1 : 0;
}

int main(int argc, char **argv) 
{ 
    foo(true);
    return 0;
}

