// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Define three pointers, which point to variables in register, shared memory and global memory respectively. The data type is the same, amp-compatible data type. Verify that they can exchange with each other. And after exchange, the value of variables by pointed by them can be changed through the pointer. </summary>

#include "../../inc/common.h"

template <typename type>
bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 99;

    vector<int> A(size);
    vector<type> G(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        G[i] = (type)1;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);
    array<type, 1> aG(e, G.begin(), rv), aG2(e, rv);

    parallel_for_each(aA.get_extent().tile<1>(), [&](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static type ts, ts2;
        type l, l2;

        type *pG = NULL;
        type *pts = NULL;
        type *pl = NULL;

        ts = (type)2;
        l = (type)3;

        if (aFlag[0] == 0)
        {
            pG = &aG[idx];
            pts = &ts;
            pl = &l;
        } else
        {
            pG = &aG2[idx];
            pts = &ts2;
            pl = &l;
        }

        type tmp = *pl;

        *pl = *pG;
        *pG = *pts;
        *pts = tmp;

        if (!Equal(*pG, (type) 2) || !Equal(*pl, (type) 1) || !Equal(*pts, (type) 3))
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

    passed = test<int>(rv) && test<unsigned int>(rv) && test<long>(rv) && test<unsigned long>(rv)
        && test<float>(rv) && test<double>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

