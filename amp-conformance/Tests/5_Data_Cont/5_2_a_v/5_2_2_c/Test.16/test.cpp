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
/// <summary>Test that a const array_view can be copy constructed from a read-write array_view</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int size = 100;

    vector<int> vec_rw(size), vec_ro(size);
    for(int i = 0; i < size; i++)
    {
        vec_rw[i] = i;
        vec_ro[i] = i;
    }

    array_view<int, 1> av_base(size, vec_ro); // rw array_view
    array_view<const int, 1> av_ro(av_base); // copy construct  ro array_view from rw array_view

    array_view<int, 1> av_rw(size, vec_rw); // for verification

    if(av_ro.get_extent()[0] != av_base.get_extent()[0]) // verify extent
    {
        printf("array_view extent different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av_ro.get_extent(), [=] (index<1> idx) __GPU
    {
        av_rw[idx] = idx[0] + 1;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av_rw[0]);

    // verify data
    for(int i = 0; i < size; i++)
    {
        if(av_rw[i] != (i + 1) || vec_rw[i] != (i + 1))
        {
            printf("data copied to vector doesnt contained updated data at index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual av_rw: [%d] Actual vec: [%d]\n", (i+1), av_rw[i], vec_rw[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

