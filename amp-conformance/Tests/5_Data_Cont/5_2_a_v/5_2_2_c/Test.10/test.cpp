//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Create an array_view of type const type using extent values e0 and e1, and a container in a CPU restricted function. Verify extent and data in array_view</summary>

#include <amptest.h>
#include <amptest_main.h>

#include "../../helper.h"

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

runall_result test_main()
{
    const int m = 100, n = 80;
    const int size = m * n;

    vector<int> vec1(size);
    for(int i = 0; i < size; i++) vec1[i] = i;

    vector<int> vec2(size);
    for(int i = 0; i < size; i++) vec2[i] = i;

    array_view<const int, 2> av1(m, n, vec1);
    array_view<int, 2> av2(m, n, vec2);

    if(m != av1.get_extent()[0]) // Verify extent
    {
        printf("array_view extent[0] different from extent used to initialize object. FAIL!\n");
        printf("Expected: [%d] Actual : [%d]\n", m, av1.get_extent()[0]);
        return runall_fail;
    }


    if(n != av1.get_extent()[1]) // Verify extent
    {
        printf("array_view extent[1] different from extent used to initialize object. FAIL!\n");
        printf("Expected: [%d] Actual : [%d]\n", n, av1.get_extent()[1]);
        return runall_fail;
    }

    // verify data
    if(!compare(vec1, av1))
    {
         printf("FAIL: array_view and vector data do not match\n");
         return runall_fail;
    }

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av2.get_extent(), [=] (index<2> idx) __GPU
    {
        av2[idx] = av1[idx] + 1;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av2(0,0));

    // verify data
    for(int i = 0; i < size; i++)
    {
        if(vec2[i] != i + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", i + 1, vec2[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

