// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test pad() function on tiled_extent.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

bool test() restrict(cpu,amp)
{
	extent<1> g1(25);
	auto tg1 = g1.tile<4>();
	auto ptg1 = tg1.pad();
	if (ptg1[0] != 28)
		return false;

	extent<2> g2(25,51);
	auto tg2 = g2.tile<4,8>();
	auto ptg2 = tg2.pad();
	if (ptg2[0] != 28)
		return false;
	if (ptg2[1] != 56)
		return false;

	extent<3> g3(25,51,85);
	auto tg3 = g3.tile<4,8,16>();
	auto ptg3 = tg3.pad();
	if (ptg3[0] != 28)
		return false;
	if (ptg3[1] != 56)
		return false;
	if (ptg3[2] != 96)
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
