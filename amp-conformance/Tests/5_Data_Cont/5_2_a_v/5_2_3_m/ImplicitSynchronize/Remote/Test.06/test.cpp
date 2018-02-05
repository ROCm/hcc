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
/// <summary>Create data on the CPU, write remotely (two GPUs), read locally with an overlapping view</summary>

#include "amptest/array_view_test.h"
#include <amptest.h>
#include <amptest_main.h>
#include <algorithm>
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

    ArrayViewTest<int, 2> av(extent<2>(25, 25));
    av.view()[index<2>(1, 1)] = 13;
    av.set_known_value(index<2>(1, 1), 13);
    av.view()[index<2>(21, 21)] = 14;
    av.set_known_value(index<2>(21, 21), 14);

    // read/write remotely on both GPUs
    Log(LogType::Info, true) << "Reading/Writing on accel 1" << std::endl;
    array_view<int, 2> remote1 = av.view().section(extent<2>(5, 5));
    parallel_for_each(accel1.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        remote1(0, 0) = remote1(1, 1);
        remote1(1, 1) = 12;
    });
    av.set_known_value(index<2>(0, 0), 13);
    av.set_known_value(index<2>(1, 1), 12);

    Log(LogType::Info, true) << "Reading/Writing on accel 2" << std::endl;
    array_view<int, 2> remote2 = av.view().section(index<2>(20, 20), extent<2>(5, 5));
    parallel_for_each(accel2.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        remote2(0, 0) = remote2(1, 1);
        remote2(1, 1) = 15;
    });
    av.set_known_value(index<2>(20, 20), 14);
    av.set_known_value(index<2>(21, 21), 15);

    return
        av.view()(0, 0) == 13 &&
        av.view()(1, 1) == 12 &&
        av.view()(20, 20) == 14 &&
        av.view()(21, 21) == 15
        ? av.pass() : av.fail();

}
