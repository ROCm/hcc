// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test deference after ++, -- operator. </summary>

#include "../../inc/common.h"

template <typename type>
bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<int> A(size);
    vector<type> G(size * 10);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        G[i * 10 + 0] = 0; G[i * 10 + 1] = 1; G[i * 10 + 2] = 2; G[i * 10 + 3] = 3; G[i * 10 + 4] = 4;
        G[i * 10 + 5] = 5; G[i * 10 + 6] = 6; G[i * 10 + 7] = 7; G[i * 10 + 8] = 8; G[i * 10 + 9] = 9;
    }

    extent<1> e(size);
    extent<1> eG(size * 10);

    array<int, 1> aA(e, A.begin(), rv);
    array<type, 1> aG(eG, G.begin(), rv), aG2(e, rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        type *p = NULL;

        if (aFlag[0] == 0)
            p = &aG[idx * 10 + 0];
        else
            p = &aG[idx * 10 + 1];

        p++;

        type *p2 = p;

        if (!Equal(*p2, (type)1))
            aA[idx] = 1;

        p2--;

        type *p3 = p2;

        if (!Equal(*p3, (type)0))
            aA[idx] = 1;

        p = &aG[idx * 10];

        if (aFlag[0] == 0)
            p2 = &aG[idx * 10 + 9];
        else
            p2 = &aG[idx * 10 + 8];

        int diff = p2 - p;

        if ((diff != 9) || (diff != (&aG[9] - &aG[0])))
            aA[idx] = 1;

        p2 = p + 9;

        if (!Equal(*p2, (type)9))
            aA[idx] = 1;

        *(p2 - 1) = 0;
    });

    A = aA;
    G = aG;

    for (int i =  0; i < size; i++)
    {
        if ((A[i] != INIT_VALUE) || !Equal(G[i * 10 + 8], (type)0))
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

