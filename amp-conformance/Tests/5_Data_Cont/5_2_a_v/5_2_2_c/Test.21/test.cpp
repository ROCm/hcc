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
/// <summary>Test that a array_view<const int, N> can be created can be created from a const int*</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int size = 100;

    int rw_data[size];
    for(int i = 0; i < size; i++) rw_data[i] = i;

    int ro_data[size];
    for(int i = 0; i < size; i++) ro_data[i] = i;
    const int* data = ro_data; // const int *

    array_view<const int, 1> av1(size, data);
    array_view<int, 1> av2(size, rw_data); // for verification

    if(size != av1.get_extent()[0]) // Verify extent
    {
        printf("Incorrect array_view extent Expected: [%d] Actual: [%d]. FAIL!\n", size, av1.get_extent()[0]);
        return runall_fail;
    }

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av2.get_extent(), [=] (index<1> idx) __GPU
    {
        av2[idx] = idx[0] + 1;
    });

    for(int i = 0; i < size; i++)
    {
        if(av2[i] != (av1[i] + 1))
        {
            printf("av2 is not updated as expected at index : %d. FAIL!\n", i);
            printf("Expected: [%d] Actual: [%d]", av1[i] + 1, av2[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}
