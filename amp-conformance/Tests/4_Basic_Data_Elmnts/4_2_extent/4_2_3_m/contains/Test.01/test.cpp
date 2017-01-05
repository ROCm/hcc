// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that contains returns true for indices within the extent space (i.e. in {0, internal, maximum}^N set).</summary>

#define NOMINMAX
#include <amptest.h>
#include <amptest_main.h>
#include <amptest/coordinates.h>
#include <limits>

using namespace concurrency;
using namespace concurrency::Test;

runall_result test(int int_max) restrict(cpu,amp)
{
	// Tests for extent rank 1
	{
		extent<1> ext(10);
		if(!(  ext.contains(index<1>(0))
			&& ext.contains(index<1>(4))
			&& ext.contains(index<1>(9))
			))
			return runall_fail;
	}
	{
		extent<1> ext(int_max);
		if(!(  ext.contains(index<1>(0))
			&& ext.contains(index<1>(int_max / 2))
			&& ext.contains(index<1>(int_max - 1))
			))
			return runall_fail;
	}

	// Tests for extent rank 2
	{
		extent<2> ext(11, 8);
		if(!(  ext.contains(index<2>(0, 0))
			&& ext.contains(index<2>(0, 3))
			&& ext.contains(index<2>(0, 7))
			&& ext.contains(index<2>(1, 0))
			&& ext.contains(index<2>(3, 5))
			&& ext.contains(index<2>(5, 7))
			&& ext.contains(index<2>(10, 0))
			&& ext.contains(index<2>(10, 3))
			&& ext.contains(index<2>(10, 7))
			))
			return runall_fail;
	}
	{
		extent<2> ext(int_max, int_max);
		if(!(  ext.contains(index<2>(0, int_max - 1))
			&& ext.contains(index<2>(int_max - 1, int_max - 1))
			&& ext.contains(index<2>(int_max - 1, int_max / 2))
			))
			return runall_fail;
	}

	// Tests for extent rank 3
	{
		extent<3> ext(2, 2, 2);
		for(int i = 0; i < 2; i++)
		{
			for(int j = 0; j < 2; j++)
			{
				for(int k = 0; k < 2; k++)
				{
					if(!ext.contains(index<3>(i, j, k)))
						return runall_fail;
				}
			}
		}
	}
	{
		extent<3> ext(3, 3, 3);
		if(!ext.contains(index<3>(1, 1, 1)))
			return runall_fail;
	}
	{
		extent<3> ext(int_max, int_max, int_max);
		if(!(  ext.contains(index<3>(0, int_max - 1, 0))
			&& ext.contains(index<3>(int_max - 1, int_max - 1, int_max - 1))
			))
			return runall_fail;
	}

	// Tests for extent rank N = 5
	{
		extent<5> ext = make_extent(1, 2, 3, 4, 5);
		if(!(  ext.contains(make_index(0, 0, 0, 0, 0))
			&& ext.contains(make_index(0, 0, 0, 0, 3))
			&& ext.contains(make_index(0, 0, 0, 0, 4))
			&& ext.contains(make_index(0, 0, 1, 0, 0))
			&& ext.contains(make_index(0, 0, 1, 2, 0))
			&& ext.contains(make_index(0, 0, 1, 3, 0))
			&& ext.contains(make_index(0, 1, 2, 0, 0))
			&& ext.contains(make_index(0, 1, 2, 0, 3))
			&& ext.contains(make_index(0, 1, 2, 0, 4))
			&& ext.contains(make_index(0, 1, 2, 3, 4))
			))
			return runall_fail;
	}
	{
		extent<5> ext = make_extent(int_max, int_max, int_max, int_max, int_max);
		if(!(  ext.contains(make_index(0, 0, 0, 0, 0))
			&& ext.contains(make_index(int_max - 1, int_max - 1, int_max - 1, int_max - 1, int_max - 1))
			))
			return runall_fail;
	}

    return runall_pass;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	int int_max = std::numeric_limits<int>::max();

	runall_result cpu_result = test(int_max);
	Log(LogType::Info, true) << "Test " << cpu_result << " on host\n";

	runall_result amp_result = GPU_INVOKE(av, runall_result, test, int_max);
	Log(LogType::Info, true) << "Test " << amp_result << " on device\n";

	return cpu_result & amp_result;
}
