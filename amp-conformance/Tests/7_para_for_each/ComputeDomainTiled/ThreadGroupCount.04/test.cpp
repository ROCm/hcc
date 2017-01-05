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
/// <summary>Test that invoking tiled parallel_for_each with maximum number of tiles per dimension (65535) succeeds. Test for 2D.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

template <int D0, int D1>
bool test(const accelerator_view& av, const tiled_extent<D0, D1>& ext, int expected)
{
	// Note: This test is only veryfing that the correct number of parallel activites
	// were created for each row/column of tiles in the domain. It is not covered
	// whether they were correctly tiled or indexed.

	std::vector<int> vec(65535);
	array_view<int, 1> vec_view(static_cast<int>(vec.size()), vec);

	parallel_for_each(av, ext, [=](tiled_index<D0, D1> tidx) restrict(amp)
	{
		if(ext[0] > ext[1])
		{
			atomic_fetch_inc(&vec_view[tidx.tile[0]]);
		}
		else
		{
			atomic_fetch_inc(&vec_view[tidx.tile[1]]);
		}
	});

	vec_view.synchronize();
	return VerifyAllSameValue(vec, expected) == -1;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;

	result &= REPORT_RESULT(test(av,
		extent<2>(65535, 1).tile<1, 1>(),
		1
		));
	result &= REPORT_RESULT(test(av,
		extent<2>(8, 2*65535).tile<2, 2>(),
		8*2
		));
	result &= REPORT_RESULT(test(av,
		extent<2>(3*65535, 4).tile<3, 2>(),
		3*4
		));

	return result;
}
