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
/// <summary>Verifies a 1d section of structs and a 1d section of floats do not overlap</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>
#include <amptest_main.h>
#include <algorithm>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct Foo
{
    float r, b, g;
};

runall_result test_main()
{
    accelerator accel = require_device(device_flags::NOT_SPECIFIED);

    if(accel.get_supports_cpu_shared_memory())
    {
        accel.set_default_cpu_access_type(ACCESS_TYPE);
    }

    std::vector<float> v(30);
    std::fill(v.begin(), v.end(), 5.0f);
    ArrayViewTest<float, 1> av(extent<1>(static_cast<int>(v.size())), v);

    Log(LogType::Info, true) << "Creating a section [0-15) and reinterpreting as structs" << std::endl;
    array_view<Foo, 1> av_struct = av.view().section(extent<1>(15)).reinterpret_as<Foo>();

    Log(LogType::Info, true) << "Creating a section [15-30)" << std::endl;
    array_view<float, 1> av_float = av.view().section(index<1>(15), extent<1>(15));

    Log(LogType::Info, true) << "Updating struct data on the GPU" << std::endl;
    parallel_for_each(accel.get_default_view(), extent<1>(1), [=](index<1>) __GPU {
        av_struct[0].r = 1.0;
        av_struct[0].b = 2.0;
        av_struct[0].g = 3.0;
    });
    av.set_known_value(index<1>(0), 1.0);
    av.set_known_value(index<1>(1), 2.0);
    av.set_known_value(index<1>(2), 3.0);

    Log(LogType::Info, true) << "Now performing implic synch on elements [15-30)" << std::endl;
    av_float[0];

    Log(LogType::Info, true) << "Now updating the struct part [0-15)" << std::endl;
    return
        av_struct[0].r == 1.0 &&
        av_struct[0].b == 2.0 &&
        av_struct[0].g == 3.0
        ? av.pass() : av.fail();
}
