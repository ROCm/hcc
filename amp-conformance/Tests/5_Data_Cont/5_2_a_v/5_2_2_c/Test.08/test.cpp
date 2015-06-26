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
/// <summary>- Create an array_view using extent value, e0, e1 and e2, and a container in a CPU restricted function. </summary>

#include <amptest.h>
#include <vector>
#include <algorithm>
#include "../../helper.h"

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int m = 2, n = 80, o = 10;
    const int size = m * n * o;

    vector<int> vec(size);
    Fill<int>(vec.data(), size);
    vector<int> vec_copy(vec);

    array_view<int, 3> av(m, n, o, vec);

    if(m != av.get_extent()[0]) // Verify extent
    {
        printf("array_view extent[0] different from extent used to initialize object. FAIL!\n");
        printf("Expected: [%d] Actual : [%d]\n", m, av.get_extent()[0]);
        return runall_fail;
    }


    if(n != av.get_extent()[1]) // Verify extent
    {
        printf("array_view extent[1] different from extent used to initialize object. FAIL!\n");
        printf("Expected: [%d] Actual : [%d]\n", n, av.get_extent()[1]);
        return runall_fail;
    }

    if(o != av.get_extent()[2]) // Verify extent
    {
        printf("array_view extent[2] different from extent used to initialize object. FAIL!\n");
        printf("Expected: [%d] Actual : [%d]\n", o, av.get_extent()[2]);
        return runall_fail;
    }

    // verify data
    if(!compare(vec, av))
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
    parallel_for_each(acc_view, av.get_extent(), [=] (index<3> idx) __GPU
    {
        av[idx]++;
    });

    // vec should be updated after this
    printf("Accessing first element of array_view [%d] to force synchronize.\n", av[0][0][0]);

    // verify data
    for(int i = 0; i < av.get_extent()[0]; i++)
    {
        for(int j = 0; j < av.get_extent()[1]; j++)
        {
            for(int k = 0; k < av.get_extent()[2]; k++)
            {
                auto expected = vec_copy[i * av.get_extent()[1] * av.get_extent()[2] + j * av.get_extent()[2] + k] + 1;
                auto actual = av(i,j,k);

                if(expected != actual)
                {
                    printf("Incorrect data. Expected [%d] Actual: [%d] FAIL!\n", expected, actual);
                    return false;
                }
             }
        }
    }


    printf("PASS!\n");
    return runall_pass;
}

