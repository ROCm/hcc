// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Convert 0 to a amp–compatible type pointer</summary>

#include "../../inc/common.h"

class s
{
public:
    int i;
    double d;
    unsigned long ul;
    float f;
};

bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);
    vector<s> G1(size);
    vector<int> G2(size);
    vector<double> G3(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        G1[i].i = 2;
        G1[i].d = 2;
        G1[i].ul = 2;
        G1[i].f = 2;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);
    array<s, 1> aG1(e, G1.begin(), rv);
    array<int, 1> aG2(e, G2.begin(), rv);
    array<double, 1> aG3(e, G3.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        s o;

        o.i = 2;
        o.d = 2;
        o.ul = 2;
        o.f = 2;

        const s o2 = o;

        const s* ps = 0;
        ps = &o2;

        if (!Equal(ps->i, 2) || !Equal(ps->d, (double)2) || !Equal(ps->ul, (unsigned long)2) || !Equal(ps->f, (float)2))
            aA[idx] = 1;

        const int i1 = 1;
        const int *pi1 = &i1;
        const double d1 = 1;
        const double *pd1 = &d1;

        if (!Equal(*pi1, (int)1) || !Equal(*pd1, (double)1))
            aA[idx] = 1;
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

