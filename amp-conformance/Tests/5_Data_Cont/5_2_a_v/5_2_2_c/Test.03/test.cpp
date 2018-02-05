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
/// <summary>Test that a readonly array_view can be constructed from an AMP array in a CPU restricted function</summary>

#include "../../helper.h"
#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int size = 100;

    vector<int> vec(size);
    Fill<int>(vec.data(), size);

    extent<1> ex(size);
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    array<int, 1> arr(ex, vec.begin(), acc_view);

    array_view<const int, 1> av1(arr);
    array_view<int, 1> av2(arr);

    if(arr.get_extent() != av1.get_extent()) // Verify extent
    {
        printf("array and array_view extents do not match. FAIL!\n");
        return runall_fail;
    }

    // verify data
    if(!compare(vec, av1))
    {
         printf("FAIL: vector not updated with array_view data\n");
         return runall_fail;
    }

    // use in parallel_for_each
    parallel_for_each(arr.get_extent(), [=] (index<1> idx) __GPU
    {
        av2[idx]++;
    });

    //verify that the data has been incremented by one
    for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); i++)
    {
        if(av1[i] != vec[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec[i] + 1, av1[i]);
            return runall_fail;
        }

        if(av2[i] != vec[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec[i] + 1, av2[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

