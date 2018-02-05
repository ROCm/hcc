// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test deference after ++, -- operator. test structure</summary>

#include "../../inc/common.h"

#ifndef FLT_EPSILON
#define FLT_EPSILON     0x1.0p-23f
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON     0x1.0p-52
#endif

struct s
{
    int i;
    double d;
    unsigned long ul;
    float f;
};

bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent().tile<1>(), [&](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static s arr[10];

        arr[0].i = 0; arr[1].i = 1; arr[2].i = 2; arr[3].i = 3; arr[4].i = 4;
        arr[0].d = 0; arr[1].d = 1; arr[2].d = 2; arr[3].d = 3; arr[4].d = 4;
        arr[0].ul = 0; arr[1].ul = 1; arr[2].ul = 2; arr[3].ul = 3; arr[4].ul = 4;
        arr[0].f = 0; arr[1].f = 1; arr[2].f = 2; arr[3].f = 3; arr[4].f = 4;
        arr[5].i = 5; arr[6].i = 6; arr[7].i = 7; arr[8].i = 8; arr[9].i = 9;
        arr[5].d = 5; arr[6].d = 6; arr[7].d = 7; arr[8].d = 8; arr[9].d = 9;
        arr[5].ul = 5; arr[6].ul = 6; arr[7].ul = 7; arr[8].ul = 8; arr[9].ul = 9;
        arr[5].f = 5; arr[6].f = 6; arr[7].f = 7; arr[8].f = 8; arr[9].f = 9;

        s *p = &arr[9];

        p--;

        s *p2 = p;

        if ((p2->i != 8) || (precise_math::fabs(p2->d - 8) > DBL_EPSILON) || (p2->ul != 8) || (precise_math::fabs(p2->f - 8) > FLT_EPSILON))
            aA[idx] = 1;

        p2++;

        s *p3 = p2;

        if ((p3->i != 9) || (precise_math::fabs(p3->d - 9) > DBL_EPSILON) || (p3->ul != 9) || (precise_math::fabs(p3->f - 9) > FLT_EPSILON))
            aA[idx] = 1;

        p = &arr[0];
        p2 = &arr[9];

        int diff = p2 - p;

        if ((diff != 9) || (diff != (&arr[9] - &arr[0])))
            aA[idx] = 1;

        p2 = p + 9;

        if ((p2->i != 9) || (precise_math::fabs(p2->d - 9) > DBL_EPSILON) || (p2->ul != 9) || (precise_math::fabs(p2->f - 9) > FLT_EPSILON))
            aA[idx] = 1;
    });

    A =aA;

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

