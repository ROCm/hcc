// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Aliasing (Use reference.)</summary>

#include "../../inc/common.h"

class c
{
public:
    int32_t i;
    double d;
    uint32_t ui;
    float f;
};

union u
{
    float f;
    double d;
};

bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);
    vector<c> Gc(size);
    vector<u> Gu(size);
    vector<int> Gi(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        Gc[i].i = 1;
        Gc[i].d = 1;
        Gc[i].ui = 1;
        Gc[i].f = 1;
        Gu[i].d = 3;
        Gu[i].f = 1;
        Gi[i] = 1;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);
    array<c, 1> aGc(e, Gc.begin(), rv);
    array<u, 1> aGu(e, Gu.begin(), rv);
    array<int, 1> aGi(e, Gi.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        c &p1 = aGc[idx], &p2 = aGc[idx];

        if (!Equal(p1.i, (int)1) || !Equal(p1.ui, (uint32_t)1) || !Equal(p1.f, (float)1) || !Equal(p1.d, (double)1)
            || !Equal(p2.i, (int)1) || !Equal(p2.ui, (uint32_t)1) || !Equal(p2.f, (float)1) || !Equal(p2.d, (double)1))
            aA[idx] = 1;

        p1.i = 2;
        p1.d = 2;
        p1.ui = 2;
        p1.f = 2;

        if (!Equal(p2.i, (int)2) || !Equal(p2.ui, (uint32_t)2) || !Equal(p2.f, (float)2) || !Equal(p2.d, (double)2))
            aA[idx] = 1;

        u &p3 = aGu[idx], &p4 = aGu[idx];

        p3.d = 2;

        if (!Equal(p4.d, (double)2) )
            aA[idx] = 1;

        int32_t &pi1 = aGi[idx], &pi2 = aGi[idx];
        pi1 = 2;
        if (pi2 != 2)
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

