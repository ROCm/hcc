// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copy from Array View to staging Array</summary>

#include "./../../CopyTestFlow.h"
#include <amptest_main.h>
#include <tuple>

using namespace Concurrency;
using namespace Concurrency::Test;

void print_access_type_tuple(std::tuple<access_type>& tup)
{
	Log(LogType::Info, true) << "CPU Access Types: (" << std::get<0>(tup) << ")" << std::endl;
}

class ArrayViewToStagingArrayTests
{
private:
	accelerator cpu_acc;
	accelerator gpu_acc;

	std::vector<std::tuple<access_type>> access_types_vec;

public:
	ArrayViewToStagingArrayTests()
	{
		cpu_acc = accelerator(accelerator::cpu_accelerator);
		gpu_acc = require_device_for<DATA_TYPE>(device_flags::NOT_SPECIFIED, false);

		if(gpu_acc.get_supports_cpu_shared_memory())
		{
			WLog(LogType::Info, true) << "Accelerator " << gpu_acc.get_description() << " supports zero copy" << std::endl;

			// Set the default cpu access type for this accelerator
			gpu_acc.set_default_cpu_access_type(DEF_ACCESS_TYPE);

			access_types_vec.push_back(std::make_tuple(access_type_none));
			access_types_vec.push_back(std::make_tuple(access_type_read));
			access_types_vec.push_back(std::make_tuple(access_type_write));
			access_types_vec.push_back(std::make_tuple(access_type_read_write));
		}
		else
		{
			access_types_vec.push_back(std::make_tuple(access_type_auto));
		}
	}

	runall_result CpuAccViewToCpuAccView()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view arr_av = cpu_acc.get_default_view();
		accelerator_view stg_arr_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayViewToStagingArray<DATA_TYPE, RANK>(cpu_av, arr_av, stg_arr_av, std::get<0>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result GpuAccViewToGpuAccView()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view arr_av = gpu_acc.get_default_view();
		accelerator_view stg_arr_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayViewToStagingArray<DATA_TYPE, RANK>(cpu_av, arr_av, stg_arr_av, std::get<0>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result CpuAccView1ToCpuAccView2()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view arr_av = cpu_acc.create_view();
		accelerator_view stg_arr_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayViewToStagingArray<DATA_TYPE, RANK>(cpu_av, arr_av, stg_arr_av, std::get<0>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}

	runall_result GpuAccView1ToGpuAccView2()
	{
		accelerator_view cpu_av = cpu_acc.get_default_view();
		accelerator_view arr_av = gpu_acc.create_view();
		accelerator_view stg_arr_av = gpu_acc.get_default_view();

		runall_result res;

		for(auto a_t_tuple : access_types_vec)
		{
			print_access_type_tuple(a_t_tuple);
			res &= CopyAndVerifyFromArrayViewToStagingArray<DATA_TYPE, RANK>(cpu_av, arr_av, stg_arr_av, std::get<0>(a_t_tuple), std::get<0>(a_t_tuple));
		}

		return res;
	}
};

runall_result test_main()
{
	ArrayViewToStagingArrayTests tests;
	runall_result res;

	res &= REPORT_RESULT(tests.CpuAccViewToCpuAccView());
	res &= REPORT_RESULT(tests.GpuAccViewToGpuAccView());
	res &= REPORT_RESULT(tests.CpuAccView1ToCpuAccView2());
	res &= REPORT_RESULT(tests.GpuAccView1ToGpuAccView2());

	return res;
}

