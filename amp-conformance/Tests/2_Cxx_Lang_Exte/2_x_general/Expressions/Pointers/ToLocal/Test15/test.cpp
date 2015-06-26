// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Define pointer which is NULL and std::nullptr_t.</summary>

#include "../../inc/common.h"

bool test(accelerator_view &rv)
{
    const int size = 1024;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);
    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        void * p = NULL;

        if (p != nullptr)
            aA[idx] = 1;

    });

    A = aA;

    for(int i = 0; i < size; i++)
    {
        if (A[i] == 1)
            return false;
    }

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

    passed = test(rv);
    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

