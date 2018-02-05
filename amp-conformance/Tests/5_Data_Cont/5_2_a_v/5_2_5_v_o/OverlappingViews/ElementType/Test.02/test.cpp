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
/// <summary>Verifies a 2d section of structs and a 3d section of uints overlap</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>
#include <amptest_main.h>
#include <algorithm>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct Foo
{
    int a, b;
};

runall_result test_main()
{
    accelerator accel = require_device(device_flags::NOT_SPECIFIED);

    if(accel.get_supports_cpu_shared_memory())
    {
        accel.set_default_cpu_access_type(ACCESS_TYPE);
    }

    std::vector<int> v(30);
    std::fill(v.begin(), v.end(), 5);
    ArrayViewTest<int, 1> av(extent<1>(static_cast<int>(v.size())), v);

    Log(LogType::Info, true) << "Creating a section [0-16) and reinterpreting as structs" << std::endl;
    array_view<Foo, 2> av_struct = av.view().section(extent<1>(16)).reinterpret_as<Foo>().view_as(extent<2>(2, 4));

    Log(LogType::Info, true) << "Creating a section [14-30) of 3d uint" << std::endl;
    array_view<unsigned int, 3> av_uint = av.view()
                                            .section(index<1>(14), extent<1>(16))
                                            .reinterpret_as<unsigned int>()
                                            .view_as(extent<3>(2, 2, 4));

    Log(LogType::Info, true) << "Updating struct data on the GPU" << std::endl;
    parallel_for_each(accel.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        av_struct[index<2>(1, 3)].a = 1;
        av_struct[index<2>(1, 3)].b = 2;
    });
    av.set_known_value(index<1>(14), 1);
    av.set_known_value(index<1>(15), 2);

    Log(LogType::Info, true) << "Now performing implic synch on elements [14-30)" << std::endl;
    av_uint[index<3>()];

    Log(LogType::Info, true) << "Elements [14-16) of the underlying data should have changed" << std::endl;

    if (av.data()[14] != 1 || av.data()[15] != 2)
    {
        Log(LogType::Error, true) << "Underlying data was not updated when it shouldn't have been" << std::endl;
        Log(LogType::Error, true) << "av.data()[14] = " << av.data()[14] << std::endl;
        Log(LogType::Error, true) << "av.data()[15] = " << av.data()[15] << std::endl;
        return runall_fail;
    }

    return
        av_uint[index<3>(0, 0, 0)] == 1 &&
        av_uint[index<3>(0, 0, 1)] == 2 &&
        av_struct[index<2>(1, 3)].a == 1 &&
        av_struct[index<2>(1, 3)].b == 2
        ? av.pass() : av.fail();
}
