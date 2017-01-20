// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test pad() function on tiled_extent</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <climits>

using namespace concurrency;
using namespace concurrency::Test;

bool Test1() restrict(cpu,amp)
{
	extent<1> ext(16);
	auto      tiled_ext = ext.tile<4>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == 16) ? true : false;
}

bool Test2() restrict(cpu,amp)
{
	extent<2> ext(16, 40);
	auto      tiled_ext = ext.tile<4, 8>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == 16 && padded_tiled_ext[1] == 40) ? true : false;
}

bool Test3() restrict(cpu,amp)
{
	extent<3> ext(16, 40, 60);
	auto      tiled_ext = ext.tile<4, 8, 12>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == 16 && padded_tiled_ext[1] == 40 && padded_tiled_ext[2] == 60) ? true : false;
}

bool Test11() restrict(cpu,amp)
{
	extent<1> ext(INT_MAX-1);
	auto      tiled_ext = ext.tile<INT_MAX-1>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == (INT_MAX-1)) ? true : false;
}

bool Test21() restrict(cpu,amp)
{
	extent<2> ext(INT_MAX-1, INT_MAX-11);
	auto      tiled_ext = ext.tile<INT_MAX-1, INT_MAX-11>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == (INT_MAX-1) && padded_tiled_ext[1] == (INT_MAX-11)) ? true : false;
}

bool Test31() restrict(cpu,amp)
{
	extent<3> ext(INT_MAX-1, INT_MAX-11, INT_MAX-111);
	auto      tiled_ext = ext.tile<INT_MAX-1, INT_MAX-11, INT_MAX-111>();
	auto      padded_tiled_ext = tiled_ext.pad();

	return (padded_tiled_ext[0] == (INT_MAX-1) && padded_tiled_ext[1] == (INT_MAX-11) && padded_tiled_ext[2] == (INT_MAX-111)) ? true : false;
}

runall_result test_main()
{
    runall_result result;

	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test1);
    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test2);
    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test3);
    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test11);
    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test21);
    result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, Test31);

	return result;
}

