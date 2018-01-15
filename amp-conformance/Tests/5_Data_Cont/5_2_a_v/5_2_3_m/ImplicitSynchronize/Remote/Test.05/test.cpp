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
/// <summary>Create data on the CPU, read remotely (two GPUs)</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    accelerator accel1 = require_device_for<int>(device_flags::NOT_SPECIFIED, false);
    accelerator accel2 = require_device_for<int>(accel1, device_flags::NOT_SPECIFIED, false);

    if(accel1.get_supports_cpu_shared_memory())
    {
        accel1.set_default_cpu_access_type(DEF_ACCESS_TYPE1);
    }

    if(accel2.get_supports_cpu_shared_memory())
    {
        accel2.set_default_cpu_access_type(DEF_ACCESS_TYPE2);
    }

    std::vector<int> v(25 * 25);
    Fill(v);
    array_view<int, 2> av(extent<2>(25, 25), v);

    // read remotely on both GPUs
    std::vector<int> result_1v(5 *5);
    array_view<int, 2> result_1(extent<2>(5, 5), result_1v);
    parallel_for_each(accel1.get_default_view(), result_1.get_extent(), [=](index<2> i) __GPU {
        result_1[i] = av[i];
    });

    std::vector<int> result_2v(5 *5);
    array_view<int, 2> result_2(extent<2>(5, 5), result_2v);
    parallel_for_each(accel2.get_default_view(), result_2.get_extent(), [=](index<2> i) __GPU {
        result_2[i] = av[i + index<2>(20, 20)]; // read a different section
    });

	auto avsec1 = av.section(extent<2>(5, 5));
	auto avsec2 = av.section(index<2>(20, 20), extent<2>(5, 5));
    return
        VerifyDataOnCpu(avsec1, result_1) &&
        VerifyDataOnCpu(avsec2, result_2)
        ? runall_pass : runall_fail;
}
