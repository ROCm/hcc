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
    int one = 1;
    int two = 2;

    parallel_for_each(aA.get_extent().tile<1>(), [&, one, two](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static c o1;

        o1.i = one;
        o1.d = 1;
        o1.ui = one;
        o1.f = one;

        c &p1 = o1, &p2 = o1;

        if (!Equal(p1.i, (int)one) || !Equal(p1.ui, (uint32_t)one) || !Equal(p1.f, (float)one) || !Equal(p1.d, (double)1)
            || !Equal(p2.i, (int)one) || !Equal(p2.ui, (uint32_t)one) || !Equal(p2.f, (float)one) || !Equal(p2.d, (double)1))
            aA[idx] = 1;

        p1.i = two;
        p1.d = 2;
        p1.ui = two;
        p1.f = two;

        if (!Equal(p2.i, (int)two) || !Equal(p2.ui, (uint32_t)two) || !Equal(p2.f, (float)two) || !Equal(p2.d, (double)2))
            aA[idx] = 1;

        tile_static u o3;

        o3.d = 1;
        o3.f = one;

        u &p3 = o3, &p4 = o3;

        p3.d = 2;

        if (!Equal(p4.d, (double)2) )
            aA[idx] = 1;

        tile_static bool b1;
        b1 = true;
        bool &pb1 = b1, &pb2 = b1;

        pb1 = false;

        if (pb2)
            aA[idx] = 1;

        tile_static uint32_t i1;
        i1 = 1;
        uint32_t &pi1 = i1, &pi2 = i1;
        pi1 = 2;
        if (pi2 != 2)
            aA[idx] = 1;

        tile_static c oa[10];

        oa[9].i = one;
        oa[9].d = 1;
        oa[9].ui = one;
        oa[9].f = two;

        c &p11 = oa[9];
        c &p21 = oa[9];

        p11.i = two;
        p11.d = 2;
        p11.ui = two;
        p11.f = two;

        if (!Equal(p21.i, (int)two) || !Equal(p21.ui, (uint32_t)two) || !Equal(p21.f, (float)two) || !Equal(p21.d, (double)2))
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

