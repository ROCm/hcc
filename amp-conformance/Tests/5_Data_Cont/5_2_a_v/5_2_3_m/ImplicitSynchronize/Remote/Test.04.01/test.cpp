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
/// <summary>Create data on the GPU, write to locally and remotely in a loop</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    accelerator acc = require_device(device_flags::NOT_SPECIFIED);

    if(acc.get_supports_cpu_shared_memory())
    {
        acc.set_default_cpu_access_type(ACCESS_TYPE);
    }

    array<int, 1> a(extent<1>(100));
    array_view<int, 1> av(a);

    for (int i = 0; i < 10; i++)
    {
        if (i % 2 == 0)
        {
            // write remotely first
            array_view<int, 1> section1 = av.section(extent<1>(20));
            section1[10] = 17;

            parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
                array_view<int, 1> section2 = av.section(index<1>(10), extent<1>(10));
                section2[0] = 19;
            });
        }
        else
        {
            // write locally first
            parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
                array_view<int, 1> section2 = av.section(index<1>(10), extent<1>(10));
                section2[0] = 19;
            });

            array_view<int, 1> section1 = av.section(extent<1>(20));
            section1[10] = 17;
        }
    }

    // since the last iteration is i = 9, the result value should be 17
    return av[10] == 17 ? runall_pass : runall_fail;
}
