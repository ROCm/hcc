// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>The pointer points to two dimension non-POD array.</summary>

#include "../../inc/common.h"

class c
{
public:
    c() __GPU_ONLY {}
    ~c() __GPU_ONLY {}

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

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent().tile<1>(), [&](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static c arr[10][10], arr2[10][10];

        double di = 0.0;
        double dj = 0.0;

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                arr[i][j].i = arr[i][j].ui = arr[i][j].f = i * 10 + j;
                arr[i][j].d = di * 10.0 + dj;
                arr2[i][j].i = arr2[i][j].ui = arr2[i][j].f = i * 10 + j + 1;
                arr2[i][j].d = di * 10.0 + dj + 1;
                dj++;
            }
            di++;
            dj = 0.0;
        }

        c *p = NULL;

        if (aFlag[0] == 0)
            p = &arr[0][0];
        else
            p = &arr2[0][0];

        double tmpd = 0;
        for (int i = 0; i < 100; i++)
        {
            if (!Equal((*p).d, tmpd) || !Equal((*p).i, (int)i) || !Equal((*p).ui, (uint32_t)i) || !Equal((*p).f, (float)i))
            {
                aA[idx] = 1;
            }
            p++;
            tmpd++;
        }

        int *pi = &(arr[5][5].i);
        if (!Equal(*pi, 55))
            aA[idx] = 1;

        unsigned int *pui = &(arr[5][5].ui);
        if (!Equal(*pi, 55))
            aA[idx] = 1;

        float *pf = &(arr[5][5].f);
        if (!Equal(*pf, (float)55))
            aA[idx] = 1;

        double *pd = &(arr[5][5].d);
        if (!Equal(*pd, (double)55))
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

