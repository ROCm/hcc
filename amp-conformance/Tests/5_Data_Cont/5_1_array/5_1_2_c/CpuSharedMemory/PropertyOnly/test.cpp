// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Using array with CPU access type none on accelerator supporting zero-copy</summary>

#include "../Common.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test1()
{
	extent<2> arr_extent = CreateRandomExtent<2>(64);
    	array<float, 2> arr(arr_extent);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test2()
{
	array<int, 1> arr1(64);
	array<int, 2> arr2(64, 64);
	array<int, 3> arr3(64, 64, 64);

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test3()
{
	extent<3> arr_extent = CreateRandomExtent<3>(64);
	std::vector<int> cont(arr_extent.size(), 10);
	array<int, 3> arr(arr_extent, cont.begin());

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test4()
{
	extent<2> arr_extent = CreateRandomExtent<2>(64);
	std::vector<int> cont(arr_extent.size(), 10);
	array<int, 2> arr(arr_extent, cont.begin(), cont.end());

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test5()
{
	std::vector<int> cont1(64, 10);
	array<int, 1> arr1(64, cont1.begin());

	std::vector<int> cont2(64 * 64, 10);
	array<int, 2> arr2(64, 64, cont2.begin());

	std::vector<int> cont3(64 * 64 * 64, 10);
	array<int, 3> arr3(64, 64, 64, cont3.begin());

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test6()
{
	std::vector<int> cont1(64, 10);
	array<int, 1> arr1(64, cont1.begin(), cont1.end());

	std::vector<int> cont2(64 * 64, 10);
	array<int, 2> arr2(64, 64, cont2.begin(), cont2.end());

	std::vector<int> cont3(64 * 64 * 64, 10);
	array<int, 3> arr3(64, 64, 64, cont3.begin(), cont3.end());

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test7()
{
	extent<3> arr_extent = CreateRandomExtent<3>(64);

	array_view<int , 3> arr_v(arr_extent);
	array_view<const int, 3> arr_v_c(arr_v);
	array<int, 3> arr(arr_v_c);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test8(accelerator& device)
{
	extent<2> arr_extent = CreateRandomExtent<2>(64);

	array_view<int , 2> arr_v(arr_extent);
	array_view<const int, 2> arr_v_c(arr_v);
	array<int, 2> arr(arr_v_c, device.get_default_view(), ACCESS_TYPE);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test9(accelerator& device)
{
	extent<2> arr_extent = CreateRandomExtent<2>(64);
    array<float, 2> arr(arr_extent, device.get_default_view(), ACCESS_TYPE);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test10(accelerator& device)
{
	array<int, 1> arr1(64, device.get_default_view(), ACCESS_TYPE);
	array<int, 2> arr2(64, 64, device.get_default_view(), ACCESS_TYPE);
	array<int, 3> arr3(64, 64, 64, device.get_default_view(), ACCESS_TYPE);

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test11(accelerator& device)
{
	extent<3> arr_extent = CreateRandomExtent<3>(64);
	std::vector<int> cont(arr_extent.size(), 10);
	array<int, 3> arr(arr_extent, cont.begin(), device.get_default_view(), ACCESS_TYPE);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test12(accelerator& device)
{
	extent<2> arr_extent = CreateRandomExtent<2>(64);
	std::vector<int> cont(arr_extent.size(), 10);
	array<int, 2> arr(arr_extent, cont.begin(), cont.end(), device.get_default_view(), ACCESS_TYPE);

	return REPORT_RESULT(VerifyCpuAccessType(arr, ACCESS_TYPE));
}

runall_result test13(accelerator& device)
{
	std::vector<int> cont1(64, 10);
	array<int, 1> arr1(64, cont1.begin(), device.get_default_view(), ACCESS_TYPE);

	std::vector<int> cont2(64 * 64, 10);
	array<int, 2> arr2(64, 64, cont2.begin(), device.get_default_view(), ACCESS_TYPE);

	std::vector<int> cont3(64 * 64 * 64, 10);
	array<int, 3> arr3(64, 64, 64, cont3.begin(), device.get_default_view(), ACCESS_TYPE);

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test14(accelerator& device)
{
	std::vector<int> cont1(64, 10);
	array<int, 1> arr1(64, cont1.begin(), cont1.end(), device.get_default_view(), ACCESS_TYPE);

	std::vector<int> cont2(64 * 64, 10);
	array<int, 2> arr2(64, 64, cont2.begin(), cont2.end(), device.get_default_view(), ACCESS_TYPE);

	std::vector<int> cont3(64 * 64 * 64, 10);
	array<int, 3> arr3(64, 64, 64, cont3.begin(), cont3.end(), device.get_default_view(), ACCESS_TYPE);

	runall_result res;

	res &= REPORT_RESULT(VerifyCpuAccessType(arr1, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr2, ACCESS_TYPE));
	res &= REPORT_RESULT(VerifyCpuAccessType(arr3, ACCESS_TYPE));

	return res;
}

runall_result test_main()
{
	accelerator device = require_device(device_flags::NOT_SPECIFIED);

	if(!device.get_supports_cpu_shared_memory())
	{
		WLog(LogType::Info, true) << "The accelerator " << device.get_description() << " does not support zero copy: Skipping" << std::endl;
		return runall_skip;
	}

	accelerator(accelerator::default_accelerator).set_default_cpu_access_type(ACCESS_TYPE);

	runall_result res;

	res &= REPORT_RESULT(test1());
	res &= REPORT_RESULT(test2());
	res &= REPORT_RESULT(test3());
	res &= REPORT_RESULT(test4());
	res &= REPORT_RESULT(test5());
	res &= REPORT_RESULT(test6());
	res &= REPORT_RESULT(test7());
	res &= REPORT_RESULT(test8(device));
	res &= REPORT_RESULT(test9(device));
	res &= REPORT_RESULT(test10(device));
	res &= REPORT_RESULT(test11(device));
	res &= REPORT_RESULT(test12(device));
	res &= REPORT_RESULT(test13(device));
	res &= REPORT_RESULT(test14(device));

	return res;
}

