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
/// <summary>Test that executing parallel_for_each over tiled_extent where extent cannot be evenly divided by a tile extent throws an exception. Test for 2D.</summary>
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

	result &= REPORT_RESULT(expect_exception(av, extent<2>(36, 49).tile<19, 49>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(36, 49).tile<9, 2>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(8, 8).tile<7, 7>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(1, 1).tile<1, 2>()));
	result &= REPORT_RESULT(expect_exception(av, extent<2>(1, std::numeric_limits<int>::max()).tile<1, 2>()));

	return result;
}

