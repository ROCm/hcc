// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>test pointer operators, ++, --, >, >=, <, <=, ==, !=. []. </summary>

#include "../../inc/common.h"

template <typename type>
bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<type> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<type, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent().template tile<1>(), [&](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static type arr[10];

        arr[0] = 0; arr[1] = 1; arr[2] = 2; arr[3] = 3; arr[4] = 4;
        arr[5] = 5; arr[6] = 6; arr[7] = 7; arr[8] = 8; arr[9] = 9;

        type *p = &arr[1];

        p++;

        if (*p != 2)
            aA[idx] = 1;

        p--;

        if (*p != 1)
            aA[idx] = 1;

        type *p2 = &arr[2];

        if (!(p < p2))
            aA[idx] = 1;

        if (!(p <= p2))
            aA[idx] = 1;

        if (!(p2 > p))
            aA[idx] = 1;

        if (!(p2 >= p))
            aA[idx] = 1;

        type *p3 = &arr[1];

        if (!(p == p3))
            aA[idx] = 1;

        if (!(p != p2))
            aA[idx] = 1;

        p--;

        if (p[9] != 9)
            aA[idx] = 1;

    });

    A = aA;

    if (A[0] != INIT_VALUE)
        return false;

    return true;
}

runall_result test_main()
{
    bool passed = true;

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return runall_skip;
    }

    accelerator_view rv = device.get_default_view();

    passed = test<int>(rv) && test<unsigned long>(rv);
    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

