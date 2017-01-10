// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy between staging Array to iterator</summary>

#include "./../../CopyTestFlow.h"
#include <amptest_main.h>
#include <deque>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
	accelerator_view cpu_av = accelerator(accelerator::cpu_accelerator).get_default_view();
	accelerator gpu_acc = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);

	if(gpu_acc.get_supports_cpu_shared_memory())
	{
		WLog(LogType::Info, true) << "Accelerator " << gpu_acc.get_description() << " supports zero copy" << std::endl;

		// Set the default cpu access type for this accelerator
		gpu_acc.set_default_cpu_access_type(DEF_ACCESS_TYPE);
	}

	accelerator_view gpu_av = gpu_acc.get_default_view();

	runall_result res;
	res &= CopyAndVerifyBetweenStagingArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(cpu_av, gpu_av);

	return res;
}

