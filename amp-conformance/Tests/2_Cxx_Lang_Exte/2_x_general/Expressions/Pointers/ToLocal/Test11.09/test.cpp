// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Define pointers in kernel functions which are called by other functions which are called by forall.</summary>

#include "../../inc/common.h"

class c
{
public:
    int i;
    double d;
    unsigned long ul;
    float f;
};

struct s //empty clas
{
};

union u
{
    float f;
    double d;
};

void f3(float &pf, double &pd, int &pi, unsigned long &pul, u &pu, s &ps, c &pc) __GPU
{
    s &soi = ps;

    pu.d = 4;
    pu.f = 2;

    pc.i = 2;
    pc.d = 2;
    pc.ul = 2;
    pc.f = 2;

    pi = 2;
    pul = 2;
    pd = 2;
    pf = 2;
}

void f2(float &pf, double &pd, int &pi, unsigned long &pul, u &pu, s &ps, c &pc) __GPU
{
    f3(pf, pd, pi, pul, pu, ps, pc);
}

void f1(float &pf, double &pd, int &pi, unsigned long &pul, u &pu, s &ps, c &pc) __GPU
{
    f2(pf, pd, pi, pul, pu, ps, pc);
}

bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        c o, o2;

        o.i = 1;
        o.d = 1;
        o.ul = 1;
        o.f = 1;

        s so, so2;

        u uo, uo2;

        uo.d = 3;
        uo.f = 1;

        int i = 1, i2;
        unsigned long ul = 1, ul2;
        double d = 1, d2;
        float f = 1, f2;

        for(int cnt = 0; cnt < 10; cnt++)
        {
            if (aFlag[0] == 0)
                f1(f, d, i, ul, uo, so, o);
            else
                f1(f2, d2, i2, ul2, uo2, so2, o2);

            if (!Equal(o.i, (int)2) || !Equal(o.ul, (unsigned long)2) || !Equal(o.f, (float)2) || !Equal(o.d, (double)2)
                ||!Equal(i, (int)2) || !Equal(ul, (unsigned long)2) || !Equal(f, (float)2) || !Equal(d, (double)2)
                || !Equal(uo.f, (float)2))
                aA[idx] = 1;

            o.i = 1;
            o.d = 1;
            o.ul = 1;
            o.f = 1;

            i = 1;
            ul = 1;
            d = 1;
            f = 1;
        }
    });

    A = aA;

    for (int i =  0; i < size; i++)
    {
        if (A[i] != INIT_VALUE)
            return false;
    }

    return true;
}

runall_result test_main()
{
    bool passed = true;

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test(rv);
    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

