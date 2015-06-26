// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>The pointer points to two dimension array.</summary>

#include "../../inc/common.h"

class c
{
public:
    int32_t i;
    double d;
    uint32_t ui;
    float f;
};

bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<int> A(size);
    vector<c> G(size);
    vector<c> G2(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        G[i].i = G[i].d = G[i].ui = G[i].f = i;
        G2[i].i = G2[i].d = G2[i].ui = G2[i].f = i;
    }

    extent<1> e(size);
    extent<2> eG(10, 10);

    array<int, 1> aA(e, A.begin(), rv);
    array<c, 2> aG(eG, G.begin(), rv);
    array<c, 2> aG2(eG, G2.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        c *p = NULL;

        if (aFlag[0] == 0)
            p = &aG[0][0];
        else
            p = &aG2[0][0];

        double di = 0;
        for (int i = 0; i < 100; i++)
        {
            if (!Equal((*p).i, (int)i) || !Equal((*p).d, di) || !Equal((*p).ui, (uint32_t)i) || !Equal((*p).f, (float)i))
            {
                aA[idx] = 1;
            }
            p++;
            di++;
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

