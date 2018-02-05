// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test assignment operator on array_views without a data source</summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

/*
* Testing assignment before p_f_e execution in CPU
* Source => array_view without data source
*/
bool test1(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	std::vector<int> vecA(M * N);
	std::vector<int> vecB(M * N);
	std::generate(vecA.begin(), vecA.end(), rand);
	std::generate(vecB.begin(), vecB.end(), rand);

	extent<2> ext(M,N);
	array_view<const int, 2> arrViewA(ext, vecA);
	array_view<const int, 2> arrViewB(ext, vecB);
	array_view<int, 2> arrViewSum(ext);
	array_view<int, 2> arrViewTarget = arrViewSum; // Assignment : Source => array_view with out data source

	parallel_for_each(av, arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
	});

	// Now verify the results
	bool passed = true;
	for (size_t i = 0; i < vecA.size(); ++i) {
		if (arrViewTarget(i / N, i % N) != (vecA[i] + vecB[i])) {
			Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewTarget(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
			passed = false;
		}
	}
	result &= REPORT_RESULT(passed);
	return passed;
}

/*
* Testing assignment after p_f_e execution in CPU
* Source => array_view without data source
*/
bool test2(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	std::vector<int> vecA(M * N);
	std::vector<int> vecB(M * N);
	std::generate(vecA.begin(), vecA.end(), rand);
	std::generate(vecB.begin(), vecB.end(), rand);

	extent<2> ext(M,N);
	array_view<const int, 2> arrViewA(ext, vecA);
	array_view<const int, 2> arrViewB(ext, vecB);
	array_view<int, 2> arrViewSum(ext);

	parallel_for_each(av, arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
	});

	array_view<int, 2> arrViewTarget = arrViewSum; // Assignment after p_f_e: Source => array_view without data source ,
	// Now verify the results
	bool passed = true;
	for (size_t i = 0; i < vecA.size(); ++i) {
		if (arrViewTarget(i / N, i % N) != (vecA[i] + vecB[i])) {
			Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewTarget(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
			passed = false;
		}
	}
	result &= REPORT_RESULT(passed);
	return passed;
}

/*
* Testing assignment in CPU
* Target => array_view without data source
* Source => array_view with data source
*/
bool test3(const accelerator_view &av)
{
	const int M = 256;
	const int N = 256;
	std::vector<int> vecA(M * N);
	std::generate(vecA.begin(), vecA.end(), rand);

	array_view<int, 2> arrViewA( M , N, vecA);
	array_view<int, 2> arrViewTarget(M,N);
	arrViewTarget = arrViewA; // Assignment : Target => array_view without data source , Source => array_view with data source

	bool passed = true;
	for (size_t i = 0; i < vecA.size(); ++i) {
		if (arrViewTarget(i / N, i % N) != vecA[i]) {
			Log(LogType::Error, true) << "Actual = " << arrViewTarget(i / N, i % N) << ", Expected = " << (vecA[i]) << std::endl;
			passed = false;
		}
	}

	REPORT_RESULT(passed);
	return passed;
}

/*
* Testing assignment in GPU
* Target => array_view without data source
* Source => array_view with data source
*/
bool test4(const accelerator_view &av)
{
	const int M = 256;
	const int N = 256;

	std::vector<int> vecA(M * N);
	std::generate(vecA.begin(), vecA.end(), rand);

	array_view<int, 2> arrViewA( M , N, vecA);
	array_view<int, 2> arrViewTarget(M,N);
	array_view<int, 1> arr_compare_result(1);

	parallel_for_each(av, extent<1>(1), [=](const index<1> &idx) mutable restrict(amp) {
		arrViewTarget = arrViewA; // Assignment : Target => array_view without data source , Source => array_view with data source

		arr_compare_result[0] = 0;
		for(int i = 0; i < arrViewTarget.get_extent()[0] ; i++ )
		{
			for(int j = 0 ; j < arrViewTarget.get_extent()[1] ; j++ )
			{
				if( arrViewTarget(i,j) != arrViewA(i,j))
				{
					atomic_fetch_inc(&arr_compare_result[0]);
				}
			}
		}
	});

	// Now verify the results
	bool passed = (arr_compare_result[0] == 0);

	REPORT_RESULT(passed);
	return passed;
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

	res &= REPORT_RESULT(test1(av));
	res &= REPORT_RESULT(test2(av));
	res &= REPORT_RESULT(test3(av));
	res &= REPORT_RESULT(test4(av));
    return res;
}

