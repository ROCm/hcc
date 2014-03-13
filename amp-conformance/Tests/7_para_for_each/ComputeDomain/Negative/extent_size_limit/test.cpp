//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Test that exceeding the extent maximum size results in an exception being thrown.</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <limits>
#include "../Utils.h"

#undef max
using namespace concurrency;
using namespace concurrency::Test;

runall_result test_main()
{
	accelerator_view av = require_device().get_default_view();

	runall_result result;

	// Note extent<1> cannot exceed the limit.

	result &= REPORT_RESULT(expect_exception(av,
		extent<2>(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()),
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (4611686014132420609) exceeds the limit (4294967295)."));
	result &= REPORT_RESULT(expect_exception(av,
		extent<2>(1 << 30, 1 << 2), // 2^32
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (4294967296) exceeds the limit (4294967295)."));

	result &= REPORT_RESULT(expect_exception(av,
		extent<3>(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()),
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (at least 4611686014132420609) exceeds the limit (4294967295)."));
	result &= REPORT_RESULT(expect_exception(av,
		extent<3>(9110917, 3627949, 558080), // The product of these will result in 1024 if the overflow is not handled correctly
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (at least 33053942219233) exceeds the limit (4294967295)."));
	result &= REPORT_RESULT(expect_exception(av,
		extent<3>(1 << 15, 1 << 2, 1 << 15), // 2^32
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (4294967296) exceeds the limit (4294967295)."));

	int dimSize[128];

	std::fill(dimSize, dimSize + 31, 2);
	std::fill(dimSize + 31, dimSize + 127, 1);
	dimSize[127] = 2;
	result &= REPORT_RESULT(expect_exception(av,
		extent<128>(dimSize), // 2^32
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (4294967296) exceeds the limit (4294967295)."));

	std::fill(dimSize, dimSize + 128, std::numeric_limits<int>::max());
	result &= REPORT_RESULT(expect_exception(av,
		extent<128>(dimSize), // max^128, maximum possible compute domain to specify
		"concurrency::parallel_for_each: unsupported compute domain, The number of threads requested (at least 4611686014132420609) exceeds the limit (4294967295)."));

	return result;
}

