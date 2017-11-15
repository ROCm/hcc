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
/// <summary>discard_data with array_view</summary>

// Create an array_view on CPU and create three random section of it.
// Modify all the three section on gpu, discard their data, call synchronize().
// Do it in a loop for multiple time and verify the underlying data.

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

	std::vector<int> data(1000, 20);
	array_view<int, 3> arr_v(10, 10, 10, data);

	index<3> origin(0, 0, 0);
	extent<3> range(8, 8, 8);

	for(int n = 0; n < 100; n++)
	{
		index<3> idx1 = GetRandomIndex(origin, range);
		index<3> idx2 = GetRandomIndex(origin, range);
		index<3> idx3 = GetRandomIndex(origin, range);

		array_view<int, 3> arr_v1 = arr_v.section(idx1);
		array_view<int, 3> arr_v2 = arr_v.section(idx2);
		array_view<int, 3> arr_v3 = arr_v.section(idx3);

		parallel_for_each(extent<3>(2, 2, 2), [=] (index<3> idx) restrict(amp) {
			arr_v1[idx] = 2;
			arr_v2[idx] = 3;
			arr_v3[idx] = 4;
		});

		arr_v1.discard_data();
		arr_v2.discard_data();
		arr_v3.discard_data();

		arr_v.synchronize();
	}

	return (VerifyAllSameValue(data, 20) == -1);
}

