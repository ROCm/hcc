// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy from Array to Array View</summary>

#include "./../../../CopyTestFlow.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

class ArrayToArrayViewTests
{
private:
	accelerator cpu_acc;
	accelerator gpu_acc1;
	accelerator gpu_acc2;

	access_list access_types_vec;

public:
	ArrayToArrayViewTests()
	{
		cpu_acc = accelerator(accelerator::cpu_accelerator);
		gpu_acc1 = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);
		gpu_acc2 = require_device_for<DATA_TYPE>(gpu_acc1, device_flags::NOT_SPECIFIED, false);

		compute_access_type_list(access_types_vec, gpu_acc1, gpu_acc2, DEF_ACCESS_TYPE1, DEF_ACCESS_TYPE2);
	}

	runall_result Gpu1AccViewToGpu2AccView()
	{
		accelerator_view gpu_av1 = gpu_acc1.get_default_view();
		accelerator_view gpu_av2 = gpu_acc2.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(gpu_av1, gpu_av2, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}
};

runall_result test_main()
{
	ArrayToArrayViewTests tests;
	runall_result res;

	res &= REPORT_RESULT(tests.Gpu1AccViewToGpu2AccView());

	return res;
}

