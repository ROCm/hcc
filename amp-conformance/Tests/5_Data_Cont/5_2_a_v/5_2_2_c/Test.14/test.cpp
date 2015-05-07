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
/// changing data using one view and make sure the other view can see the update - in a gpu restricted function</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;


int main()
{
    const int size = 100;

    vector<int> vec(size), results_vec(size);
    Fill<int>(vec.data(), size);

    array_view<int, 1> av1(size, vec);
    array_view<int, 1> result(size, results_vec);

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    parallel_for_each(acc_view, av1.get_extent(), [=](index<1> idx) restrict(amp) {

        array_view<int, 1> av2(av1); // copy construct

        if(av1.get_extent()[0] != av2.get_extent()[0]) // Verify extent
        {
            result[idx] = 11;
            return;
        }

        // verify data
        if(av1[idx] != av2[idx])
        {
            result[idx] = 55;
            return;
        }

        // update data
        av1[idx] = av1[idx] + 1;

        // av2 should be updated after this - verify data
        if(av1[idx] != av2[idx])
        {
            result[idx] = 66;
            return;
        }

        result[idx] = 100;
    });

    result.synchronize();

    // check that all tests passed on the restricted function
    for(int i = 0; i < size;i++)
    {
        if(results_vec[i] != 100)
        {
            printf("Fail: Test %d failed for index %d\n", results_vec[i], i);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

