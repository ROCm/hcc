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
/// <summary>Test that an Rank 2, float array_view can be constructed from an AMP array in a cpu restricted function</summary>

#include "../../helper.h"
#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int xsize = 10;
    const int ysize = 10;
    const int size = xsize * ysize;

    vector<float> vec(size);
    Fill<float>(vec.data(), size);
    vector<float> vec_copy(vec);

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    extent<2> ex(xsize, ysize);
    array<float, 2> arr(ex, vec.begin(), acc_view);

    array_view<float, 2> av(arr);

    if(arr.get_extent() != av.get_extent()) // verify extent
    {
        printf("array and array_view extents do not match. FAIL!\n");
        return runall_fail;
    }

    // verify data
    if(!compare(vec, av))
    {
         printf("FAIL: array_view and vector data do not match\n");
         return runall_fail;
    }

    // use in parallel_for_each
    parallel_for_each(arr.get_extent(), [=] (index<2> idx) __GPU
    {
        av[idx]++;
    });

    // arr should automatically be updated at this point
    vec = arr;

    // verify data
    if(!compare(vec, av))
    {
         printf("FAIL: vector not updated with array_view data\n");
         return runall_fail;
    }

    //verify that the data has been incremented by one
    for(size_t i = 0; i < vec.size(); i++)
    {
        if(vec[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%f] Actual: [%f] FAIL!\n", vec_copy[i] + 1, vec[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

