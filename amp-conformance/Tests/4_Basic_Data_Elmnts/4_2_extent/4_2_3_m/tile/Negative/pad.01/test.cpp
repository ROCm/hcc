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
#include <string>
#include <climits>

using namespace Concurrency;
using namespace Concurrency::Test;

bool Test1()
{
	extent<1> ext(INT_MAX);
	auto      tiled_ext = ext.tile<4>();
	auto      padded_tiled_ext = tiled_ext.pad();

	try
	{
		array<int, 1> ar(padded_tiled_ext);
	}
	catch (const runtime_exception &e)
	{
		return true;
	}

	return false;
}

bool Test2()
{
	extent<2> ext(INT_MAX, INT_MAX);
	auto      tiled_ext = ext.tile<4,4>();
	auto      padded_tiled_ext = tiled_ext.pad();

	try
	{
		array<int, 2> ar(padded_tiled_ext);
	}
	catch (const runtime_exception &e)
	{
		return true;
	}

	return false;
}

bool Test3()
{
	extent<3> ext(INT_MAX, INT_MAX, INT_MAX);
	auto      tiled_ext = ext.tile<4,4,4>();
	auto      padded_tiled_ext = tiled_ext.pad();

	try
	{
		array<int, 3> ar(padded_tiled_ext);
	}
	catch (const runtime_exception &e)
	{
		return true;
	}

	return false;
}

runall_result test_main()
{
    runall_result result;
    result &= REPORT_RESULT(Test1());
    result &= REPORT_RESULT(Test2());
    result &= REPORT_RESULT(Test3());
    return result;
}

