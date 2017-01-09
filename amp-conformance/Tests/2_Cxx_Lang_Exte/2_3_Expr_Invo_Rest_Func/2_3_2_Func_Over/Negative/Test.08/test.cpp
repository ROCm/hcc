// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Cannot call amp from cpu</summary>
//#Expects: Error: error C3930

#include <amptest.h>

class c2
{
public:
    int f(int) restrict(amp)
    {
        return 1;
    }

    int f(float) restrict(amp)
    {
        return 0;
    }
};

class c1
{
public:
    int b(int) restrict(cpu)
    {
        c2 o;

        int i;

        return o.f(i);
    }
};

bool test()
{
    c1 o;

    int i = 0;

    int flag = o.b(i);

    return ((flag == 1) ? true : false);
}

int main(int argc, char **argv)
{
    int ret = test();

    printf("failed\n");

    return 1;
}

