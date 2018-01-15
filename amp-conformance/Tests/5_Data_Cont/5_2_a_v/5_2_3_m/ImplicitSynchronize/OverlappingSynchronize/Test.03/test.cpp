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
/// <summary>Create two overlapping array_view(s) on top of another array. Capture and modify all of them in p_f_e. Verify synchronization works correctly.</summary>

#include <amptest.h>
#include <amptest_main.h>

#define DATA_SIZE (256 * 3)

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
	accelerator gpuDevice = require_device(device_flags::NOT_SPECIFIED);

	if(gpuDevice.get_supports_cpu_shared_memory())
	{
		gpuDevice.set_default_cpu_access_type(ACCESS_TYPE);
	}

	std::vector<int> data(DATA_SIZE, 2);
	array<int, 1> dataArray(DATA_SIZE, data.begin(), gpuDevice.get_default_view());

	// Create two overlapping array views.
	array_view<int, 1> arrayView1 = dataArray.section(0, 2 * DATA_SIZE/3);
	array_view<int, 1> arrayView2 = dataArray.section(DATA_SIZE/3, 2 * DATA_SIZE/3);

	parallel_for_each(arrayView1.get_extent(), [=, &dataArray](index<1> idx) restrict(amp) {
        atomic_fetch_add(&(arrayView1(idx)), 1);
		atomic_fetch_add(&(dataArray(idx)), 1);
        atomic_fetch_add(&(arrayView2(idx)), 1);
    });

	arrayView1.synchronize();
	arrayView2.synchronize();

	data = dataArray;

	for(int i = 0; i < DATA_SIZE/3; i++)
	{
		if(data[i] != 4)
		{
			Log(LogType::Error, true) << "Incorrect result at (" << i << ") - expected " << 4 << " but got " << data[i] << std::endl;
			return runall_fail;
		}
	}

	for(int i = DATA_SIZE/3; i < 2 * DATA_SIZE/3; i++)
	{
		if(data[i] != 5)
		{
			Log(LogType::Error, true) << "Incorrect result at (" << i << ") - expected " << 5 << " but got " << data[i] << std::endl;
			return runall_fail;
		}
	}

	for(int i = 2 * DATA_SIZE/3; i < DATA_SIZE; i++)
	{
		if(data[i] != 3)
		{
			Log(LogType::Error, true) << "Incorrect result at (" << i << ") - expected " << 3 << " but got " << data[i] << std::endl;
			return runall_fail;
		}
	}

	return runall_pass;
}
