// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test copy construct of array_view using array_view without a data source</summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

/*
* Test copy construct of array_view object using array_view having no data source , before p_f_e
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
	array_view<int, 2> arrViewTarget(arrViewSum); // Copy Constructing

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
* Test copy construct of array_view object using array_view having no data source , after p_f_e
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

	array_view<int, 2> arrViewTarget(arrViewSum); // Copy Constructing
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

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

	res &= REPORT_RESULT(test1(av));
	res &= REPORT_RESULT(test2(av));

    return res;
}

