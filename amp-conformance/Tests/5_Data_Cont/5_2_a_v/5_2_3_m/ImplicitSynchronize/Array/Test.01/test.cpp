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
/// <summary>
/// Create an array_view on the CPU, write on the GPU, copy construct an array and verify pending writes are
/// copied directly from cached copy on GPU and the underlying data on cpu is unaffected.
/// </summary>

#include <amptest/array_view_test.h>
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

	accelerator_view av = acc.get_default_view();

    ArrayViewTest<int, 1> arr_v(extent<1>(10));

    Log(LogType::Info, true) << "Writing on the GPU" << std::endl;
    array_view<int, 1> gpu_view = arr_v.view();
    parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(amp) {
        gpu_view(0) = 17;
    });

    Log(LogType::Info, true) << "Copying to array" << std::endl;
    array<int, 1> a(arr_v.view(), av);

	std::vector<int> results_v(1);
    array_view<int, 1> results(1, results_v);

    parallel_for_each(av, extent<1>(1), [=, &a](index<1>) restrict(amp) {
        results[0] = a[0];
    });

    int result = results[0];
    Log(LogType::Info, true) << "Result is: " << result << " Expected: 17" << std::endl;
    return result == 17 ? arr_v.pass() : arr_v.fail();
}

