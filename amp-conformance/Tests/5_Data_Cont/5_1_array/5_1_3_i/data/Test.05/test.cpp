// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests that accessing the array indexer is valid on the host if the array is stored on the cpu_accelerator. Note, this is not a data correctness test and thus no initialization or verification of the retrieved value is performed. Associated bug: 243794</summary>

#include <amptest.h>
#include <stdio.h>

using namespace concurrency;
using namespace concurrency::Test;


#define MAX_LEN 10

bool test_feature(void)
{
	// Get the accelerator for the CPU
	accelerator device(accelerator::cpu_accelerator);
	accelerator_view av = device.get_default_view();

	// Create a device array to store the results
	array<float, 1> dResult(MAX_LEN, av);

	// Read directly from the array
	float pxData = 0;
	index<1> idx(0);
	// We shouldn't get an exception here because we're running on the cpu and the array is on the cpu too
	pxData = dResult[idx];
	// Note, this test only cares that the indexing operator doesn't throw an exception.
	// Therefore I don't care or verify what the retrieved value is.

	return true;
}

int main()
{
	bool passed = true;

	passed &= test_feature();

	Log(LogType::Info, true) << (passed ? "Passed!" : "Failed!") << std::endl;
    return passed ? 0 : 1;
}

