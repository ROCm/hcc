// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy from Array to Array View</summary>

#include "./../../CopyTestFlow.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

class ArrayToArrayViewTests
{
private:
	accelerator cpu_acc;
	accelerator gpu_acc;

	access_list access_types_vec;

public:
	ArrayToArrayViewTests()
	{
		cpu_acc = accelerator(accelerator::cpu_accelerator);
		gpu_acc = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);

		compute_access_type_list(access_types_vec, gpu_acc, DEF_ACCESS_TYPE);
	}

	runall_result CpuAccViewToCpuAccView()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(cpu_av, cpu_av, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result GpuAccViewToGpuAccView()
	{
		accelerator_view gpu_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(gpu_av, gpu_av, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result CpuAccView1ToCpuAccView2()
	{
		accelerator_view cpu_av1 = cpu_acc.create_view();
		accelerator_view cpu_av2 = cpu_acc.create_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(cpu_av1, cpu_av2, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result GpuAccView1ToGpuAccView2()
	{
		accelerator_view gpu_av1 = gpu_acc.create_view();
		accelerator_view gpu_av2 = gpu_acc.create_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(gpu_av1, gpu_av2, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result CpuAccViewToGpuAccView()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view gpu_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(cpu_av, gpu_av, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result GpuAccViewToCpuAccView()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view gpu_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayToArrayView<DATA_TYPE, RANK>(gpu_av, cpu_av, std::get<0>(a_t_tuple), std::get<1>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}
};

runall_result test_main()
{
	ArrayToArrayViewTests tests;
	runall_result res;

	res &= REPORT_RESULT(tests.CpuAccViewToCpuAccView());
	res &= REPORT_RESULT(tests.GpuAccViewToGpuAccView());
	res &= REPORT_RESULT(tests.CpuAccView1ToCpuAccView2());
	res &= REPORT_RESULT(tests.GpuAccView1ToGpuAccView2());
	res &= REPORT_RESULT(tests.CpuAccViewToGpuAccView());
	res &= REPORT_RESULT(tests.GpuAccViewToCpuAccView());

	return res;
}

