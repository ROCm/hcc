// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test the compound type. Change the value </summary>

#include "../../inc/common.h"

template <typename type>
struct s
{
    type a;
    type b;
    type c;
    type d;
};

template <typename type>
bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<int> A(size);
    vector<s<type>> G(size * 10);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;

        G[i * 10 + 0].a=0;G[i * 10 + 0].b=0;G[i * 10 + 0].c=0;G[i * 10 + 0].d=0;
        G[i * 10 + 1].a=1;G[i * 10 + 1].b=1;G[i * 10 + 1].c=1;G[i * 10 + 1].d=1;
        G[i * 10 + 2].a=2;G[i * 10 + 2].b=2;G[i * 10 + 2].c=2;G[i * 10 + 2].d=2;
        G[i * 10 + 3].a=3;G[i * 10 + 3].b=3;G[i * 10 + 3].c=3;G[i * 10 + 3].d=3;
        G[i * 10 + 4].a=4;G[i * 10 + 4].b=4;G[i * 10 + 4].c=4;G[i * 10 + 4].d=4;
        G[i * 10 + 5].a=5;G[i * 10 + 5].b=5;G[i * 10 + 5].c=5;G[i * 10 + 5].d=5;
        G[i * 10 + 6].a=6;G[i * 10 + 6].b=6;G[i * 10 + 6].c=6;G[i * 10 + 6].d=6;
        G[i * 10 + 7].a=7;G[i * 10 + 7].b=7;G[i * 10 + 7].c=7;G[i * 10 + 7].d=7;
        G[i * 10 + 8].a=8;G[i * 10 + 8].b=8;G[i * 10 + 8].c=8;G[i * 10 + 8].d=8;
        G[i * 10 + 9].a=9;G[i * 10 + 9].b=9;G[i * 10 + 9].c=9;G[i * 10 + 9].d=9;
    }

    extent<1> e(size);
    extent<1> eg(size * 10);
    array<int, 1> aA(e, A.begin(), rv);
    array<s<type>, 1> aG(eg, G.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        type *p = NULL;
        if (aFlag[0] == 0)
            p = &(aG[idx[0] * 10].a);
        else
            p = &(aG[idx[0] * 10 + 1].a);

        if (!Equal(*p++, (type)0) || !Equal(*p++, (type)0) || !Equal(*p++, (type)0) || !Equal(*p++, (type)0))
            aA[idx] = 1;

        if (!Equal(*p, (type)1))
            aA[idx] = 1;

        if (aFlag[0] == 0)
            p = &(aG[idx[0] * 10 + 9].a);
        else
            p = &(aG[idx[0] * 10 + 1].a);

        *p = 10;
        *(p + 1) = 10;
        *(p + 2) = 10;
        *(p + 3) = 10;

        p = &(aG[idx[0] * 10 + 8].a);

        if (!Equal(*p++, (type)8) || !Equal(*p++, (type)8) || !Equal(*p++, (type)8) || !Equal(*p++, (type)8)
            || !Equal(*p++, (type)10) || !Equal(*p++, (type)10) || !Equal(*p++, (type)10) || !Equal(*p++, (type)10))
            aA[idx] = 1;
    });

    A = aA;
    G = aG;

    for (int i =  0; i < size; i++)
    {
        if (A[i] != INIT_VALUE)
            return false;
        if (!Equal(G[i * 10 + 9].a, (type)10) || !Equal(G[i * 10 + 9].b, (type)10) ||
            !Equal(G[i * 10 + 9].c, (type)10) || !Equal(G[i * 10 + 9].d, (type)10))
            return false;
    }

    return true;
}

runall_result test_main()
{
    bool passed = true;

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test<int>(rv) && test<unsigned long>(rv) && test<float>(rv) && test<double>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

