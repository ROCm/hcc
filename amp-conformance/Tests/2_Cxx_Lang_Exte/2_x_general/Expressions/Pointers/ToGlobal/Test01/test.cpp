// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>The pointer points to another global variable. Change the value </summary>

#include "../../inc/common.h"

template <typename type>
bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<type> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = (type)1;
    }

    extent<1> e(size);

    array<type, 1> aA(e, A.begin(), rv), aA2(e, rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        type *p = NULL;

        if (aFlag[0] == 0)
            p = &aA[idx];
        else
            p = &aA2[idx];

        *p = (type)2;
    });

    A = aA;

    for (int i =  0; i < size; i++)
    {
        if (!Equal(A[i], (type)2))
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

