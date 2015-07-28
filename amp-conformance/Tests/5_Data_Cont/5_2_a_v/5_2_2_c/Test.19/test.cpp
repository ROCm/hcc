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
/// <summary>Test that a read-write array_view<int, N> can be created can be created from int*</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int size = 100;

    int* data = new int[size];
    Fill<int>(data, size);

    array_view<int, 1> av1(size, data);
    array_view<int, 1> av2(av1); // copy construct

    if(av1.get_extent()[0] != av2.get_extent()[0]) // Verify extent
    {
        printf("array_view extent different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    // verify data
    for(int i = 0; i < size; i++)
    {
        if(data[i] != av1[i])
        {
            printf("array_view data does not match original data at index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual : [%d]\n", data[i], av1[i]);
            return runall_fail;
        }
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
    for(int i = 0; i < size; i++)
    {
        if(data[i] != av1[i])
        {
            printf("data copied to vector doesnt contained updated data at index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual : [%d]\n", data[i], av1[i]);
            return runall_fail;
        }
    }

    delete[] data;
    printf("PASS!\n");
    return runall_pass;
}

