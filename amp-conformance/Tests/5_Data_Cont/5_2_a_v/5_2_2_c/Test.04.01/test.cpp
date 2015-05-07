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
/// <summary>Test that an array_view can be constructed from a T* in a CPU restricted function</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view acc_view = device.get_default_view();

    const int size = 100;

    vector<int> vec(size);
    Fill<int>(vec.data(), size);
    vector<int> vec_copy(vec);

    extent<1> ex(size);
    array_view<int, 1> av(ex, vec.data());

    if(ex != av.get_extent()) // Verify extent
    {
        printf("array_view extent different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    if(av.get_extent() != ex) // Verify extent
    {
        printf("array_view extent is different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    // verify data
    if(!Verify(vec.data(), av.data(), vec.size()))
    {
        printf("array_view data does not match original data. FAIL!\n");
        return runall_fail;
    }

    // use in parallel_for_each
    parallel_for_each(acc_view, av.get_extent(), [=] (index<1> idx) __GPU
    {
        av[idx]++;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av[0]);

    // verify data
    for(int i = 0; i < size; i++)
    {
        if(av[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec_copy[i] + 1, av[i]);
            return runall_fail;
        }

        if(vec[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec_copy[i] + 1, vec[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

