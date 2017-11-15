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
/// <summary>Create data on the GPU, write remotely (CPU and GPU), read locally with an overlapping view</summary>

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

    Log(LogType::Info, true) << "Creating array on accel 1" << std::endl;
    array<int, 2> a(extent<2>(25, 25), accel1.get_default_view());
    array_view<int, 2> av(a);

    Log(LogType::Info, true) << "Setting some data on accel 1" << std::endl;
    parallel_for_each(accel1.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        av[index<2>(1, 1)] = 13;
        av[index<2>(21, 21)] = 14;
    });

    Log(LogType::Info, true) << "Reading/Writing on the CPU" << std::endl;
    array_view<int, 2> remote1 = av.section(extent<2>(5, 5));
    remote1(0, 0) = remote1(1, 1);
    remote1(1, 1) = 12;

    Log(LogType::Info, true) << "Reading/Writing on accel 2" << std::endl;
    array_view<int, 2> remote2 = av.section(index<2>(20, 20), extent<2>(5, 5));
    parallel_for_each(accel2.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        remote2(0, 0) = remote2(1, 1);
        remote2(1, 1) = 15;
    });

    Log(LogType::Info, true) << "Reading results on accel 1" << std::endl;
    std::vector<int> results_v(4);
    array_view<int, 1> results(4, results_v);
    parallel_for_each(accel1.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        results[0] = av(0, 0);
        results[1] = av(1, 1);
        results[2] = av(20, 20);
        results[3] = av(21, 21);
    });

    return
        // this verifies the data on the CPU as a "Remote" read
        results[0] == 13 && results[1] == 12 && results[2] == 14 && results[3] == 15
        ? runall_pass : runall_fail;

}
