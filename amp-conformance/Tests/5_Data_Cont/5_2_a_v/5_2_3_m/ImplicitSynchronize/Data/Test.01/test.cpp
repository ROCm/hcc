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
/// <summary>Create a view on the GPU, write on the GPU, call data() on the CPU and verify a synch</summary>

#include <amptest/array_view_test.h>
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

    array<int, 1> a(extent<1>(10));

    Log(LogType::Info, true) << "Writing on the GPU" << std::endl;
    parallel_for_each(extent<1>(1), [&a](index<1>) __GPU {
        a(0) = 17;
    });

    array_view<int, 1> av(a);
    int result = av.data()[0];
    Log(LogType::Info, true) << "Result is: " << result << " Expected: 17" << std::endl;
    return result == 17 ? runall_pass : runall_fail;
}
