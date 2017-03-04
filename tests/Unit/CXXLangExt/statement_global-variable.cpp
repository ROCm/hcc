
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

[[hc]] int flag;

void foo(bool set) restrict(amp, cpu)
{
    flag = set ? 1 : 0;
}

int main(int argc, char **argv) 
{ 
    foo(true);
    return 0;
}

