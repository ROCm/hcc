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
/// <summary>Verifies a 2d section of structs and a 3d section of structs overlap</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>
#include <amptest_main.h>
#include <algorithm>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct d2
{
    int a, b;
};

struct d3
{
    int a, b, c;
};

runall_result test_main()
{
    accelerator accel = require_device(device_flags::NOT_SPECIFIED);

    if(accel.get_supports_cpu_shared_memory())
    {
        accel.set_default_cpu_access_type(ACCESS_TYPE);
    }

    std::vector<int> v(100);
    std::fill(v.begin(), v.end(), 5);
    ArrayViewTest<int, 1> av(extent<1>(static_cast<int>(v.size())), v);

    Log(LogType::Info, true) << "Creating a section [0-54) and reinterpreting as 3d-structs" << std::endl;
    array_view<d3, 3> av_struct_remote = av.view().section(extent<1>(54)).reinterpret_as<d3>().view_as(extent<3>(2, 2, 3));

    Log(LogType::Info, true) << "Creating a section [2-10) of 2d-structs" << std::endl;
    array_view<d2, 2> av_struct_local = av.view().section(index<1>(2), extent<1>(8))
                                            .reinterpret_as<d2>()
                                            .view_as(extent<2>(2, 2));

    Log(LogType::Info, true) << "Updating struct data on the GPU" << std::endl;
    parallel_for_each(accel.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        av_struct_remote[index<3>(0, 0, 0)].a = 1;
        av_struct_remote[index<3>(0, 0, 0)].b = 2;
        av_struct_remote[index<3>(0, 0, 0)].c = 3;
    });
    av.set_known_value(index<1>(0), 1);
    av.set_known_value(index<1>(1), 2);
    av.set_known_value(index<1>(2), 3);

    Log(LogType::Info, true) << "Now performing implic synch on elements [2-10)" << std::endl;
    av_struct_local[index<2>()];

    Log(LogType::Info, true) << "Element 2 of the underlying data should have changed" << std::endl;

    if (av.data()[2] != 3)
    {
        Log(LogType::Error, true) << "Underlying data was not updated when it shouldn't have been" << std::endl;
        Log(LogType::Error, true) << "data[2] = " << av.data()[2] << std::endl;
        return runall_fail;
    }

    return
        av_struct_remote[index<3>(0, 0, 0)].a == 1 &&
        av_struct_remote[index<3>(0, 0, 0)].b == 2 &&
        av_struct_remote[index<3>(0, 0, 0)].c == 3 &&
        av_struct_local[index<2>(0, 0)].a == 3 &&
        av_struct_local[index<2>(0, 0)].b == 5
        ? av.pass() : av.fail();


}
