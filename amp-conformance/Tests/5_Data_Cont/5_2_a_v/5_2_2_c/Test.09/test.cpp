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
/// <summary>Create an array_view of const type using a single extent value, e0, and a container in a CPU restricted function</summary>
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
    for(int i = 0; i < size; i++) vec[i] = i;
    vector<int> vec_copy(vec);

    array_view<const int, 1> av1(size, vec);
    array_view<int, 1> av2(size, vec); // non const array to check updates to data in av1

    if(size != av1.get_extent()[0]) // Verify extent
    {
        printf("array_view extent different from extent used to initialize object. FAIL!\n");
        return runall_fail;
    }

    // verify data
    if(!equal(vec.begin(), vec.end(), av1.data()))
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
    parallel_for_each(acc_view, av2.get_extent(), [=] (index<1> idx) __GPU
    {
        av2[idx] = idx[0] + 1;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av2[0]);

    // verify data
    for(unsigned int i = 0; i < static_cast<unsigned int>(vec.size()); i++)
    {
        if(vec[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec_copy[i] + 1, vec[i]);
            return runall_fail;
        }

        if(av1[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec_copy[i] + 1, av1[i]);
            return runall_fail;
        }

        if(av2[i] != vec_copy[i] + 1)
        {
            printf("Incorrect updated data. Expected [%d] Actual: [%d] FAIL!\n", vec_copy[i] + 1, av2[i]);
            return runall_fail;
        }
    }

    printf("PASS!\n");
    return runall_pass;
}

