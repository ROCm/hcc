// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that extent with 0 size in any dimension does not contain any indices.</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <amptest/coordinates.h>

using namespace concurrency;
using namespace concurrency::Test;

runall_result test() restrict(cpu,amp)
{
	// Tests for extent rank 1
	{
		extent<1> ext(0);
		if(!(  !ext.contains(index<1>(-1))
			&& !ext.contains(index<1>(0))
			&& !ext.contains(index<1>(1))
			))
			return runall_fail;
	}

	// Tests for extent rank 2
	{
		extent<2> ext(11, 0);
		if(!(  !ext.contains(index<2>(0, 0))
			&& !ext.contains(index<2>(5, 0))
			&& !ext.contains(index<2>(5, 1))
			&& !ext.contains(index<2>(0, 5))
			))
			return runall_fail;
	}
	{
		extent<2> ext(0, 8);
		if(!(  !ext.contains(index<2>(0, 0))
			&& !ext.contains(index<2>(0, 1))
			&& !ext.contains(index<2>(-1, 1))
			&& !ext.contains(index<2>(2, 5))
			))
			return runall_fail;
	}

	// Tests for extent rank 3
	{
		extent<3> ext(0, 3, 4);
		if(!(  !ext.contains(index<3>(0, 0, 0))
			&& !ext.contains(index<3>(0, 2, 3))
			&& !ext.contains(index<3>(-1, 1, 0))
			))
			return runall_fail;
	}
	{
		extent<3> ext(5, 0, 4);
		if(!(  !ext.contains(index<3>(0, 0, 0))
			&& !ext.contains(index<3>(0, -1, 3))
			&& !ext.contains(index<3>(-1, 1, 0))
			))
			return runall_fail;
	}
	{
		extent<3> ext(5, 3, 0);
		if(!(  !ext.contains(index<3>(0, 0, 0))
			&& !ext.contains(index<3>(0, 3, -1))
			&& !ext.contains(index<3>(-1, 0, 1))
			))
			return runall_fail;
	}
	{
		extent<3> ext(0, 0, 0);
		if(!(  !ext.contains(index<3>(0, 0, 0))
			))
			return runall_fail;
	}

	// Tests for extent rank N = 5
	{
		extent<5> ext = make_extent(0, 2, 3, 4, 5);
		if(!(  !ext.contains(make_index(0, 0, 0, 0, 0))
			&& !ext.contains(make_index(-1, 0, 0, 1, 0))
			&& !ext.contains(make_index(1, 0, 0, 0, 1))
			))
			return runall_fail;
	}
	{
		extent<5> ext = make_extent(1, 0, 3, 4, 5);
		if(!(  !ext.contains(make_index(0, 0, 0, 0, 0))
			&& !ext.contains(make_index(0, -1, 0, 1, 0))
			&& !ext.contains(make_index(0, 1, 0, 0, 1))
			))
			return runall_fail;
	}
	{
		extent<5> ext = make_extent(1, 2, 3, 4, 0);
		if(!(  !ext.contains(make_index(0, 0, 0, 0, 0))
			&& !ext.contains(make_index(0, 0, 0, 1, -1))
			&& !ext.contains(make_index(0, 1, 0, 0, 1))
			))
			return runall_fail;
	}
	{
		extent<5> ext = make_extent(0, 0, 0, 0, 0);
		if(!(  !ext.contains(make_index(0, 0, 0, 0, 0))
			))
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

