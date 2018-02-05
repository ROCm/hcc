// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy from staging Array to staging Array</summary>

#include "./../../CopyTestFlow.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

class StagingArrayToStagingArrayTests
{
private:
	accelerator cpu_acc;
	accelerator gpu_acc;

public:
	StagingArrayToStagingArrayTests()
	{
		cpu_acc = accelerator(accelerator::cpu_accelerator);
		gpu_acc = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);

		if(gpu_acc.get_supports_cpu_shared_memory())
		{
			WLog(LogType::Info, true) << "Accelerator " << gpu_acc.get_description() << " supports zero copy" << std::endl;

			// Set the default cpu access type for this accelerator
			gpu_acc.set_default_cpu_access_type(DEF_ACCESS_TYPE);
		}
	}

	bool CpuAccViewToCpuAccView()
	{
		accelerator_view cpu_av1 = cpu_acc.get_default_view();
		accelerator_view stg_arr_av1 = gpu_acc.get_default_view();

		accelerator_view cpu_av2 = cpu_acc.get_default_view();
		accelerator_view stg_arr_av2 = gpu_acc.get_default_view();

		return CopyAndVerifyFromStagingArrayToStagingArray<DATA_TYPE, RANK>(cpu_av1, stg_arr_av1, cpu_av2, stg_arr_av2);
	}

	bool CpuAccView1ToCpuAccView2()
	{
		accelerator_view cpu_av1 = cpu_acc.get_default_view();
		accelerator_view stg_arr_av1 = gpu_acc.get_default_view();

		accelerator_view cpu_av2 = cpu_acc.create_view();
		accelerator_view stg_arr_av2 = gpu_acc.get_default_view();

		return CopyAndVerifyFromStagingArrayToStagingArray<DATA_TYPE, RANK>(cpu_av1, stg_arr_av1, cpu_av2, stg_arr_av2);
	}

	bool GpuAccView1ToGpuAccView2()
	{
		accelerator_view cpu_av1 = cpu_acc.get_default_view();
		accelerator_view stg_arr_av1 = gpu_acc.get_default_view();

		accelerator_view cpu_av2 = cpu_acc.get_default_view();
		accelerator_view stg_arr_av2 = gpu_acc.create_view();

		return CopyAndVerifyFromStagingArrayToStagingArray<DATA_TYPE, RANK>(cpu_av1, stg_arr_av1, cpu_av2, stg_arr_av2);
	}
};

runall_result test_main()
{
	StagingArrayToStagingArrayTests tests;
	runall_result res;

	res &= REPORT_RESULT(tests.CpuAccViewToCpuAccView());
	res &= REPORT_RESULT(tests.CpuAccView1ToCpuAccView2());
	res &= REPORT_RESULT(tests.GpuAccView1ToGpuAccView2());

	return res;
}

