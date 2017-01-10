// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests that using the array indexer on the host when the array is NOT stored on the cpu_accelerator will throw a runtime_excepton.</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <stdio.h>

using namespace concurrency;
using namespace concurrency::Test;

#define MAX_LEN 10

bool test_feature(void)
{
    accelerator device = require_device(Test::Device::ALL_DEVICES);
    accelerator_view av = device.get_default_view();

	access_type arr_cpu_access_type = access_type_auto;

	if(device.get_supports_cpu_shared_memory())
	{
		arr_cpu_access_type = access_type_none;
	}

	// Create a device array to store the results
	array<float, 1> dResult(MAX_LEN, av, arr_cpu_access_type);

	// Read directly from the array on the host cpu
	float pxData = 0;
	index<1> idx(0);
	try {
		pxData = dResult[idx];	// <= expect runtime_exception
	}
	catch(runtime_exception ex) {
		Log(LogType::Info, true) << "runtime_exception occured as expected: " << ex.what() << std::endl;
		return true;
	}
	catch(...) {
		Log(LogType::Error, true) << "An unknown exception occured trying to index into array based on the cpu_accelerator. "
			<< "Expected runtime_exception but got caught something else."
			<< std::endl;
		return false;
	}

	Log(LogType::Error, true) << "A runtime_exception was expected but did not occur." << std::endl;
	return false;
}

runall_result test_main()
{
	return REPORT_RESULT(test_feature());
}

