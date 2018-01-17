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
/// <summary>Negative: compute domain size is equal zero in one more more dimensions</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <limits>
#include "../Utils.h"

#undef max
using namespace concurrency;
using namespace concurrency::Test;

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;

	result &= REPORT_RESULT(expect_exception(av, extent<1>(0)));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(0, 0)));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(0, 5)));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(16, 0)));
	result &= REPORT_RESULT(expect_exception(av, extent<3>(8, 0, 8)));

	int dimSize[128];

	std::fill(dimSize, dimSize + 128, 0);
	result &= REPORT_RESULT(expect_exception(av, extent<128>(dimSize))); // 0,...,0

	std::fill(dimSize, dimSize + 128, 2);
	dimSize[70] = 0;
	result &= REPORT_RESULT(expect_exception(av, extent<128>(dimSize))); // 2,...,2,0,2,...,2

	std::fill(dimSize, dimSize + 128, std::numeric_limits<int>::max());
	dimSize[127] = 0;
	result &= REPORT_RESULT(expect_exception(av, extent<128>(dimSize))); // max,...,max,0

	return result;
}

