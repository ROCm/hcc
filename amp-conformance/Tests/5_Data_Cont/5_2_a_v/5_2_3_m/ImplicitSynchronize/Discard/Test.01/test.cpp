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
/// <summary>Create an Array View on the CPU, discard its data by calling discard_data(), modify in p_f_e, synchronize and verify data</summary>

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

	std::vector<int> data(10, 20);
	array_view<int, 1> arr_v(10, data);

	arr_v.discard_data();

	parallel_for_each(arr_v.get_extent(), [=] (index<1> idx) restrict(amp) {
		arr_v[idx] = 5;
	});

	arr_v.synchronize();

	return (VerifyAllSameValue(data, 5) == -1);
}


