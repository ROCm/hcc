// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test const-correctness for extent and index.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::Test;

runall_result test() restrict(cpu,amp)
{
	// Note: The actual result of call has minor significance here, the point is to be able to compile.

	{
		const extent<1> ext(1);
		const index<1> idx(0);
		if(!ext.contains(idx))
			return runall_fail;
	}
	{
		extent<1> ext(1);
		const index<1> idx(0);
		if(!ext.contains(idx))
			return runall_fail;
	}

    return runall_pass;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result cpu_result = test();
	Log(LogType::Info, true) << "Test " << cpu_result << " on host\n";

	runall_result amp_result = GPU_INVOKE(av, runall_result, test);
	Log(LogType::Info, true) << "Test " << amp_result << " on device\n";

	return cpu_result & amp_result;
}

