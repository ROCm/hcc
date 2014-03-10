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
/// <summary>Test that parallel_for_each is executed for a maximum allowed compute domain size 2^32 - 1 for extent rank 2, 3.</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <amptest/coordinates.h>
using namespace concurrency;
using namespace concurrency::Test;

template<int N>
bool test(const accelerator_view& av, const extent<N>& ext)
{
	const int result_length = 131072;
	std::vector<int> result(static_cast<size_t>(result_length));
	array_view<int, 1> result_view(extent<1>(result_length), result);

	parallel_for_each(av, ext, [=] (index<N> idx) restrict(amp)
	{
		atomic_fetch_inc(&result_view[flatten(idx, ext) % result_length]);
	});

	result_view.synchronize();

	// Every but last element of the vector should be incremented 32768 times and the last one 32767 times.
	std::vector<int>::iterator it = std::find_if(result.begin(), result.end() - 1, [](int x){return x != 32768;});
	return it == result.end() - 1
		&& result[result_length - 1] == 32767;
}


runall_result test_main()
{
        // test doesn't require double support but require high end cards with high performance
        // to finish compute in less than windows timeout.
	accelerator_view av = require_device(device_flags::D3D11_GPU|device_flags::DOUBLE).get_default_view();
	
	runall_result result;

	// Note: extent<1> cannot reach the maximum domain size.

	result &= REPORT_RESULT(test(av, extent<2>(65537, 65535)    ));
	
	result &= REPORT_RESULT(test(av, extent<3>(286331153, 5, 3) ));
	
	result &= REPORT_RESULT(test(av, extent<3>(3, 5, 286331153) ));

	return result;
}
