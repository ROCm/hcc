// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Using array with CPU access type write on accelerator supporting zero-copy</summary>

#include "../Common.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
	accelerator device = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);

	if(!device.get_supports_cpu_shared_memory())
	{
		WLog(LogType::Info, true) << "The accelerator " << device.get_description() << " does not support zero copy: Skipping" << std::endl;
		return runall_skip;
	}

	extent<RANK> arr_extent = CreateRandomExtent<RANK>(256);
		array<DATA_TYPE, RANK> arr(arr_extent, device.get_default_view(), access_type_write);

	if(!VerifyCpuAccessType(arr, access_type_write)) { return runall_fail; }

	Write<DATA_TYPE, RANK>(arr, 100);

	std::vector<DATA_TYPE> vec(arr.get_extent().size());
	copy(arr, vec.begin());

	if(VerifyAllSameValue(vec, static_cast<DATA_TYPE>(100)) != -1) { return runall_fail; }

	for(int i = 1; i <= 100; i++)
	{
		parallel_for_each(device.get_default_view(), arr.get_extent(), [&](index<RANK> idx) restrict(amp)
		{
			arr[idx] += 2;
		});

		index<RANK> idx;
		for(int i = 0; i < RANK; i++) { idx[i] = arr.get_extent()[i] - 1; }

		arr[idx] = 50;
	}

	vec.clear();
	vec.resize(arr.get_extent().size());
	copy(arr, vec.begin());

	if(vec[arr.get_extent().size() - 1] != 50)
	{
		Log(LogType::Error, true) << "Last element of array has wrong value." << std::endl;
		Log(LogType::Error, true) << "Expected: 50 Actual:" << vec[vec.size() - 1] << std::endl;

		return runall_fail;
	}

	vec[arr.get_extent().size() - 1] = 300;

	return REPORT_RESULT((VerifyAllSameValue(vec, static_cast<DATA_TYPE>(300)) == -1));
}
