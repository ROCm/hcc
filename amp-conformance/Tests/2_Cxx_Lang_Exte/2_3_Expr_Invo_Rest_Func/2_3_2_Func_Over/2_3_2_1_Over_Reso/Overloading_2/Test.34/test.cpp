// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Select cpu function address over amp function address</summary>

#include <amptest.h>

class c1
{
public:
    int f(int) restrict(amp)
    {
        return 0;
    }

    int f(float) restrict(amp)
    {
        return 0;
    }

    int f(int) restrict(cpu)
    {
        return 1;
    }

    int f(float) restrict(cpu)
    {
        return 0;
    }
};

bool test()
{
    int (c1::*p)(int) restrict(cpu) = &c1::f;

    int i = 0;

    c1 o;

    int flag = (o.*p)(i);

    return ((flag == 1) ? true : false);
}

int main(int argc, char **argv)
{
    bool passed = test();

    printf("%s\n", passed ? "passed\n" : "failed");

    return passed ? 0 : 1;
}

