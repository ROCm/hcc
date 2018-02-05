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
/// <summary>Create an AV, create overlapping views and use both to write data</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    accelerator acc = require_device(device_flags::NOT_SPECIFIED);

    if(acc.get_supports_cpu_shared_memory())
    {
        acc.set_default_cpu_access_type(ACCESS_TYPE);
    }

    array<long, 1> a(extent<1>(50));
    array_view<long, 1> av(a);

    parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
        array_view<long, 1> section1 = av.section(index<1>(10), extent<1>(30));
        section1[3] = 17;
    });

    parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
        array_view<long, 1> section2 = av.section(index<1>(0), extent<1>(25));
        section2[13] = 19;
    });

    parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
        array_view<long, 1> section2 = av.section(index<1>(0), extent<1>(25));
        section2[3] = 15;
    });

    return av[13] == 19 && av[3] == 15 ? runall_pass : runall_fail;
}
