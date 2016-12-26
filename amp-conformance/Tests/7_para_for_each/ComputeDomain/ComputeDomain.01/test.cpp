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
/// <summary>Test the kernel is executed over every element of a compute domain being extent rank 1, 2, 3, 128.</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <amptest/coordinates.h>
using namespace concurrency;
using namespace concurrency::Test;

template<int N>
bool test(const accelerator_view& av, const extent<N>& ext)
{
	std::vector<int> vec(ext.size());
	array_view<int, N> vec_view(ext, vec);
	vec_view.discard_data();

	parallel_for_each(av, ext, [=] (index<N> idx) restrict(amp)
	{
		vec_view[idx] = flatten(idx, ext) + 1;
	});

	vec_view.synchronize();

	bool passed = true;
	for(int i = 0; i < static_cast<int>(vec.size()); i++)
	{
		if(vec[i] != i + 1)
		{
			Log(LogType::Error, true) << "Mismatch on index: " << i << ", actual: " << vec[i] << ", expected: " << (i + 1) << std::endl;
			passed = false;
		}
	}
	return passed;
}


runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;

	result &= REPORT_RESULT(test(av, extent<1>(1)         ));
	result &= REPORT_RESULT(test(av, extent<1>(255)       ));
	result &= REPORT_RESULT(test(av, extent<1>(1024)      ));
	result &= REPORT_RESULT(test(av, extent<1>(1025)      ));

	result &= REPORT_RESULT(test(av, extent<2>(1, 1)      ));
	result &= REPORT_RESULT(test(av, extent<2>(1, 255)    ));
	result &= REPORT_RESULT(test(av, extent<2>(32, 8)     )); //256
	result &= REPORT_RESULT(test(av, extent<2>(25, 41)    )); //1025

	result &= REPORT_RESULT(test(av, extent<3>(1, 10, 1)  ));
	result &= REPORT_RESULT(test(av, extent<3>(3, 5, 17)  )); //255
	result &= REPORT_RESULT(test(av, extent<3>(256, 2, 2) )); //1024
	result &= REPORT_RESULT(test(av, extent<3>(5, 5, 41)  )); //1025

// XXX bypass the test and make it fail directly
#if 0
	int dimSize[128];

	std::fill(dimSize, dimSize + 128, 1);
	result &= REPORT_RESULT(test(av, extent<128>(dimSize) )); // 1,1,...,1

	std::fill(dimSize, dimSize + 128, 1);
	dimSize[0] = 2;
	dimSize[33] = 3;
	dimSize[75] = 5;
	dimSize[127] = 9;
	result &= REPORT_RESULT(test(av, extent<128>(dimSize) )); // 2,1,...,1,3,1,...,1,5,1,...,1,9 (270)

	std::fill(dimSize, dimSize + 128, 1);
	std::fill(dimSize + 33, dimSize + 43, 2);
	result &= REPORT_RESULT(test(av, extent<128>(dimSize) )); // 1024
#else
        result &= false;
#endif

	return result;
}
