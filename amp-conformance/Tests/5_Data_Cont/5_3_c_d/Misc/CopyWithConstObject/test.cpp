// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary><![CDATA[Copy involving const array<T, N> and const array view<T,N> for each API]]></summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

using std::vector;

unsigned int size = 10;

runall_result Array_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	const array<int, 2> src(e, src_v.begin(), av);
	array<int, 2> dst(e);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	const array<int, 2> src(e, src_v.begin(), av);

	array<int, 2> data(e);
	const array_view<int, 2> dst(data);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_to_iter(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	const array<int, 2> src(e, src_v.begin(), av);

	vector<int> dst(e.size());

	copy(src, dst.begin());

	return VerifyDataOnCpu(src, dst);
}

runall_result Const_iter2_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);
	array<int, 2> dst(e, av);

	copy(src.cbegin(), src.cend(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Const_iter_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);
	array<int, 2> dst(e, av);

	copy(src.cbegin(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_view_const_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data(e, src_v.begin(), av);
	const array_view<const int, 2> src(data);

	array<int, 2> dst(e);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_view_const_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data1(e, src_v.begin(), av);
	const array_view<const int, 2> src(data1);

	array<int, 2> data2(e);
	const array_view<int, 2> dst(data2);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_view_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data(e, src_v.begin(), av);
	const array_view<int, 2> src(data);

	array<int, 2> dst(e);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_view_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data1(e, src_v.begin(), av);
	const array_view<int, 2> src(data1);

	array<int, 2> data2(e);
	const array_view<int, 2> dst(data2);

	copy(src, dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_view_to_iter(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data(e, src_v.begin(), av);
	const array_view<int, 2> src(data);

	vector<int> dst(e.size());

	copy(src, dst.begin());

	return VerifyDataOnCpu(src, dst);
}

runall_result Iter2_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);

	array<int, 2> data(e, av);
	const array_view<int, 2> dst(data);

	copy(src.begin(), src.end(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Iter_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);

	array<int, 2> data(e, av);
	const array_view<int, 2> dst(data);

	copy(src.begin(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Const_iter2_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);

	array<int, 2> data(e, av);
	const array_view<int, 2> dst(data);

	copy(src.cbegin(), src.cend(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Const_iter_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src(e.size(), 10);

	array<int, 2> data(e, av);
	const array_view<int, 2> dst(data);

	copy(src.cbegin(), dst);

	return VerifyDataOnCpu(src, dst);
}

runall_result Array_to_const_pointer(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	const array<int, 2> src(e, src_v.begin(), av);

	vector<int> dst_v(e.size());
	int* const dst = dst_v.data();

	copy(src, dst);

	return VerifyDataOnCpu(src, dst_v);
}

runall_result Const_pointer_to_array(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	int* const src = src_v.data();

	array<int, 2> dst(e, av);

	copy(src, dst);

	return VerifyDataOnCpu(src_v, dst);
}

runall_result Array_view_to_const_pointer(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	array<int, 2> data(e, src_v.begin(), av);
	const array_view<int, 2> src(data);

	vector<int> dst_v(e.size());
	int* const dst = dst_v.data();

	copy(src, dst);

	return VerifyDataOnCpu(src, dst_v);
}

runall_result Const_pointer_to_array_view(accelerator_view& av)
{
	extent<2> e(size, size);

    vector<int> src_v(e.size(), 10);
	int* const src = src_v.data();

	array<int, 2> data(e, av);
	const array_view<int, 2> dst(data);

	copy(src, dst);

	return VerifyDataOnCpu(src_v, dst);
}


runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;

    result &= REPORT_RESULT(Array_to_array(av));
    result &= REPORT_RESULT(Array_to_array_view(av));
    result &= REPORT_RESULT(Array_to_iter(av));
	result &= REPORT_RESULT(Const_iter2_to_array(av));
	result &= REPORT_RESULT(Const_iter_to_array(av));
	result &= REPORT_RESULT(Array_view_const_to_array(av));
	result &= REPORT_RESULT(Array_view_const_to_array_view(av));
	result &= REPORT_RESULT(Array_view_to_array(av));
	result &= REPORT_RESULT(Array_view_to_array_view(av));
	result &= REPORT_RESULT(Array_view_to_iter(av));
	result &= REPORT_RESULT(Iter2_to_array_view(av));
	result &= REPORT_RESULT(Iter_to_array_view(av));
	result &= REPORT_RESULT(Const_iter2_to_array_view(av));
	result &= REPORT_RESULT(Const_iter_to_array_view(av));
	result &= REPORT_RESULT(Array_to_const_pointer(av));
	result &= REPORT_RESULT(Const_pointer_to_array(av));
	result &= REPORT_RESULT(Array_view_to_const_pointer(av));
	result &= REPORT_RESULT(Const_pointer_to_array_view(av));

    return result;
}

