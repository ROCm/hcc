// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test truncate() function on tiled_extent.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

bool test() restrict(cpu,amp)
{
	extent<1> g1(25);
	auto tg1 = g1.tile<4>();
	auto ttg1 = tg1.truncate();
	if (ttg1[0] != 24)
		return false;

	extent<2> g2(25,51);
	auto tg2 = g2.tile<4,8>();
	auto ttg2 = tg2.truncate();
	if (ttg2[0] != 24)
		return false;
	if (ttg2[1] != 48)
		return false;

	extent<3> g3(25,51,85);
	auto tg3 = g3.tile<4,8,16>();
	auto ttg3 = tg3.truncate();
	if (ttg3[0] != 24)
		return false;
	if (ttg3[1] != 48)
		return false;
	if (ttg3[2] != 80)
		return false;

	return true;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, test);
	return result;
}
