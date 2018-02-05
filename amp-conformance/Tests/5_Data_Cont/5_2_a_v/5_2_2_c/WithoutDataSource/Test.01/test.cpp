// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test array_views without a data source with the given extent </summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

// Test array_view of given extent
bool test(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;
	const int TILE_DIM0 = 16;
	const int TILE_DIM1 = 16;

	std::vector<int> vecA(M * N);
	std::vector<int> vecB(M * N);
	std::generate(vecA.begin(), vecA.end(), rand);
	std::generate(vecB.begin(), vecB.end(), rand);

	extent<2> ext(M,N);
	array_view<const int, 2> arrViewA(ext, vecA);
	array_view<const int, 2> arrViewB(ext, vecB);
	array_view<int, 2> arrViewSum(ext);

	result &= REPORT_RESULT(arrViewSum.get_extent() == ext);
	parallel_for_each(av, arrViewSum.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
	});

	array_view<int, 2> arrViewDiff(ext);
	parallel_for_each(av, arrViewDiff.get_extent().tile<TILE_DIM0,TILE_DIM1>(), [=](const tiled_index<TILE_DIM0,TILE_DIM1> &idx) restrict(amp) {
		arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
	});

	// Now verify the results
	bool passed = true;
	for (size_t i = 0; i < vecA.size(); ++i) {
		if (arrViewSum(i / N, i % N) != (vecA[i] + vecB[i])) {
			Log(LogType::Error, true) << "Sum(" << i / N << ", " << i % N << ") = " << arrViewSum(i / N, i % N) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
			passed = false;
		}

		if (arrViewDiff(i / N, i % N) != (vecA[i] - vecB[i])) {
			Log(LogType::Error, true) << "Diff(" << i / N << ", " << i % N << ") = " << arrViewDiff(i / N, i % N) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
			passed = false;
	 	}
	}
	result &= REPORT_RESULT(passed);
	return passed;
}

// Tests 1D array_view
bool test1(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int TILE_DIM0 = 16;

	std::vector<int> vecA(M);
	std::vector<int> vecB(M);
	std::generate(vecA.begin(), vecA.end(), rand);
	std::generate(vecB.begin(), vecB.end(), rand);

	array_view<const int, 1> arrViewA(M, vecA);
	array_view<const int, 1> arrViewB(M, vecB);
	array_view<int, 1> arrViewSum(M);

	result &= REPORT_RESULT(arrViewSum.get_extent() == extent<1>(M));
	parallel_for_each(av, arrViewSum.get_extent(), [=](const index<1> &idx) restrict(amp) {
		arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
	});

	array_view<int, 1> arrViewDiff(M);
	parallel_for_each(av, arrViewDiff.get_extent().tile<TILE_DIM0>(), [=](const tiled_index<TILE_DIM0> &idx) restrict(amp) {
		arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
	});

	// Now verify the results
	bool passed = true;
	for (size_t i = 0; i < vecA.size(); ++i) {
		if (arrViewSum(i) != (vecA[i] + vecB[i])) {
			Log(LogType::Error, true) << "Sum(" << i  << ") = " << arrViewSum(i) << ", Expected = " << (vecA[i] + vecB[i]) << std::endl;
			passed = false;
		}

		if (arrViewDiff(i) != (vecA[i] - vecB[i])) {
			Log(LogType::Error, true) << "Diff(" << i << ") = " << arrViewDiff(i) << ", Expected = " << (vecA[i] - vecB[i]) << std::endl;
			passed = false;
	 	}
	}
	result &= REPORT_RESULT(passed);
	return passed;
}

// Tests 3D array_view
bool test3(const accelerator_view &av)
{
	runall_result result;
	const int M = 128;
	const int N = 128;
	const int O = 128;
	const int TILE_DIM0 = 8;
	const int TILE_DIM1 = 8;
	const int TILE_DIM2 = 8;

	std::vector<int> vecA(M * N * O);
	std::vector<int> vecB(M * N * O);
	std::generate(vecA.begin(), vecA.end(), rand);
	std::generate(vecB.begin(), vecB.end(), rand);

	array_view<const int, 3> arrViewA(M, N, O, vecA);
	array_view<const int, 3> arrViewB(M, N, O, vecB);
	array_view<int, 3> arrViewSum(M, N ,O);

	result &= REPORT_RESULT(arrViewSum.get_extent() == extent<3>(M,N,O));
	parallel_for_each(av, arrViewSum.get_extent(), [=](const index<3> &idx) restrict(amp) {
		arrViewSum[idx] = arrViewA[idx] + arrViewB[idx];
	});

	array_view<int, 3> arrViewDiff(M, N, O);
	parallel_for_each(av, arrViewDiff.get_extent().tile<TILE_DIM0,TILE_DIM1,TILE_DIM2>(), [=](const tiled_index<TILE_DIM0,TILE_DIM1,TILE_DIM2> &idx) restrict(amp) {
		arrViewDiff[idx] = arrViewA[idx] - arrViewB[idx];
	});

	// Now verify the results
	bool passed = true;

	for(size_t i = 0 ; i < M ; i++)
	{
		for(size_t j = 0 ; j < N ; j++)
		{
			for(size_t k = 0 ; k < O ; k++)
			{
				int linear_idx = i * (N * O)  + j * O + k;
				if (arrViewSum(i,j,k) != (vecA[linear_idx] + vecB[linear_idx])) {
					Log(LogType::Error, true) <<  "Sum(" << i << ", " << j << ", " << k << ") = " << arrViewSum(i,j,k) << ", Expected = " << (vecA[linear_idx] + vecB[linear_idx]) << std::endl;
					passed = false;
				}

				if (arrViewDiff(i,j,k) != (vecA[linear_idx] - vecB[linear_idx])) {
					Log(LogType::Error, true) <<  "Diff(" << i << ", " << j << ", " << k << ") = " << arrViewDiff(i,j,k) << ", Expected = " << (vecA[linear_idx] - vecB[linear_idx]) << std::endl;
					passed = false;
				}

			}
		}
	}

	result &= REPORT_RESULT(passed);
	return passed;
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

	res &= REPORT_RESULT(test(av));
    res &= REPORT_RESULT(test1(av));
	res &= REPORT_RESULT(test3(av));
    return res;
}

