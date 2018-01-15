// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Define pointers in nested scope.</summary>

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

        int i = 1 , i2;
        unsigned long ul = 1, ul2;
        double d = 1, d2;
        float f = 1, f2;

        for(int cnt = 0; cnt < 10; cnt++)
        {

            s *ps = NULL;
            u* pu = NULL;
            c* pc = NULL;
            int *pi = NULL;
            unsigned long *pul = NULL;
            double *pd = NULL;
            float *pf = NULL;

            if (aFlag[0] == 0)
            {
                ps = &so;
                pu = &uo;
                pc = &o;
                pi = &i;
                pul = &ul;
                pd = &d;
                pf = &f;
            } else
            {
                ps = &so2;
                pu = &uo2;
                pc = &o2;
                pi = &i2;
                pul = &ul2;
                pd = &d2;
                pf = &f2;
            }

            pu->d = 4;
            pu->f = 2;

            if (!Equal(uo.f, (float)2))
                aA[idx] = 1;

            u uot = *pu;

            if (!Equal(uot.f, (float)2))
                aA[idx] = 1;

            pc->i = 2;
            pc->d = 2;
            pc->ul = 2;
            pc->f = 2;

            if (!Equal(o.i, 2) || !Equal(o.ul, (unsigned long)2) || !Equal(pc->f, (float)2) || !Equal(pc->d, (double)2))
                aA[idx] = 1;

            c ot = *pc;
            c *pct = &ot;

            if (!Equal(ot.i, 2) || !Equal(ot.ul, (unsigned long)2) || !Equal(pct->f, (float)2) || !Equal(pct->d, (double)2))
                aA[idx] = 1;

            *pi = 2;
            *pul = 2;
            *pd = 2;
            *pf = 2;

            if (!Equal(i, 2) || !Equal(ul, (unsigned long)2) || !Equal(f, (float)2) || !Equal(d, (double)2))
                aA[idx] = 1;

            int it = *pi;
            unsigned long ult = *pul;
            double dt = *pd;
            float ft = *pf;

            if (!Equal(it, 2) || !Equal(ult, (unsigned long)2) || !Equal(ft, (float)2) || !Equal(dt, (double)2))
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

