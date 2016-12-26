// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy between Array and iterator</summary>

#include <amptest_main.h>
#include "./../../CopyTestFlow.h"
#include <deque>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result ArrayOnCpu()
{
	accelerator_view cpu_av = accelerator(accelerator::cpu_accelerator).get_default_view();
	return CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(cpu_av, access_type_none);
}

runall_result ArrayOnGpu()
{
	accelerator gpu_acc = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);
	accelerator_view gpu_av = gpu_acc.get_default_view();

	runall_result res;

	if(gpu_acc.get_supports_cpu_shared_memory())
	{
		Log(LogType::Info, true) << "Accelerator " <<  gpu_acc.get_description() << "supports zero copy" << std::endl;

		// Set the default cpu access type for this accelerator
		gpu_acc.set_default_cpu_access_type(DEF_ACCESS_TYPE);

		res &= REPORT_RESULT((CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(gpu_av, access_type_none)));
		res &= REPORT_RESULT((CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(gpu_av, access_type_read)));
		res &= REPORT_RESULT((CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(gpu_av, access_type_write)));
		res &= REPORT_RESULT((CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(gpu_av, access_type_read_write)));
	}
	else
	{
		res &= REPORT_RESULT((CopyAndVerifyBetweenArrayAndIterator<DATA_TYPE, RANK, STL_CONT>(gpu_av, access_type_none)));
	}

	return res;
}

runall_result test_main()
{
	runall_result res;

	res &= REPORT_RESULT(ArrayOnCpu());
	res &= REPORT_RESULT(ArrayOnGpu());

	return res;
}

