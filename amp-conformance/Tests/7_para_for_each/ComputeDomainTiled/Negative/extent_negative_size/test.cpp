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
/// <summary>Test that passing an extent with a negative size in one or more dimensions results in an expection being thrown. Test for extent rank 1, 2, 3, tile divisibly and non-divisibly.</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <limits>
#include "../Utils.h"
#undef min
using namespace concurrency;
using namespace concurrency::Test;

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;

	result &= REPORT_RESULT(expect_exception(av, extent<1>(-1).tile<1>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(-2, 3).tile<2, 1>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(-2, -2).tile<2, 2>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(2, -1).tile<32, 1>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(std::numeric_limits<int>::min(), std::numeric_limits<int>::min()).tile<3, 3>()));
	result &= REPORT_RESULT(expect_exception(av, extent<3>(8, -1, 8).tile<2, 1, 5>()));
	result &= REPORT_RESULT(expect_exception(av, extent<3>(8, 3, -2).tile<2, 1, 5>()));

	return result;
}

