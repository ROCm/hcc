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
/// <summary>Copy construct an array_view from another array_view. Ensure that a shallow copy is made by
/// changing data using one view and make sure the other view can see the update</summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

runall_result test_main()
{
    const int size = 100;

    vector<int> vec(size);
    Fill<int>(vec.data(), size);

    array_view<int, 1> av1(size, vec);
    array_view<int, 1> av2(av1); // copy construct

    if(av1.get_extent()[0] != av2.get_extent()[0]) // Verify extent
    {
        printf("array_view extent different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    // verify data
    if(!VerifyDataOnCpu(av2, vec))
    {
        printf("array_view data does not match original data. FAIL!\n");
        return runall_fail;
    }

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av1.get_extent(), [=] (index<1> idx) __GPU
    {
        av1[idx] = av1[idx] + 1;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av1[0]);

    // verify data
    if(!VerifyDataOnCpu(av2, vec))
    {
        printf("data copied to vector doesnt contained updated data. FAIL!\n");
        return runall_fail;
    }

    printf("PASS!\n");
    return runall_pass;
}

