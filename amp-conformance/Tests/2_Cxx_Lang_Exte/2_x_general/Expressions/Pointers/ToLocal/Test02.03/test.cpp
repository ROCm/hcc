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

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        s<type> arr[10];

        arr[0].a=0;arr[0].b=0;arr[0].c=0;arr[0].d=0;
        arr[1].a=1;arr[1].b=1;arr[1].c=1;arr[1].d=1;
        arr[2].a=2;arr[2].b=2;arr[2].c=2;arr[2].d=2;
        arr[3].a=3;arr[3].b=3;arr[3].c=3;arr[3].d=3;
        arr[4].a=4;arr[4].b=4;arr[4].c=4;arr[4].d=4;
        arr[5].a=5;arr[5].b=5;arr[5].c=5;arr[5].d=5;
        arr[6].a=6;arr[6].b=6;arr[6].c=6;arr[6].d=6;
        arr[7].a=7;arr[7].b=7;arr[7].c=7;arr[7].d=7;
        arr[8].a=8;arr[8].b=8;arr[8].c=8;arr[8].d=8;
        arr[9].a=9;arr[9].b=9;arr[9].c=9;arr[9].d=9;

        type *p = NULL;
        if (aFlag[0] == 0)
            p = &(arr[0].a);
        else
            p = &(arr[1].a);

        if (!Equal(*p++, (type)0) || !Equal(*p++, (type)0) || !Equal(*p++, (type)0) || !Equal(*p++, (type)0))
            aA[idx] = 1;

        if (!Equal(*p, (type)1))
            aA[idx] = 1;

        if (aFlag[0] == 0)
            p = &(arr[9].a);
        else
            p = &(arr[1].a);
        *p = 10;
        *(p + 1) = 10;
        *(p + 2) = 10;
        *(p + 3) = 10;

        p = &(arr[8].a);

        if (!Equal(*p++, (type)8) || !Equal(*p++, (type)8) || !Equal(*p++, (type)8) || !Equal(*p++, (type)8)
            || !Equal(*p++, (type)10) || !Equal(*p++, (type)10) || !Equal(*p++, (type)10) || !Equal(*p++, (type)10))
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

    passed = test<int>(rv) && test<unsigned long>(rv) && test<float>(rv) && test<double>(rv);
    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

