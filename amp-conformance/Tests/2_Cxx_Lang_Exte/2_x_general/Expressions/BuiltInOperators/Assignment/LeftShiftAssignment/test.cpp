// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test checks that the built-in left shift assignment operator works inside a vector function</summary>

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include "amptest.h"
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// Initialize the input with random data
void InitializeArray(vector <int> &vM, int range)
{
    for(int i=0; i< vM.size(); ++i) { vM[i] = rand() % range; }
}

runall_result test_main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view rv = device.get_default_view();

    const int size = 10;

    vector<int> A(size);
    InitializeArray(A, sizeof(int));
    array<int, 1> aA(size, A.begin(), A.end(), rv);

    vector<int> B(size);
    InitializeArray(B, INT_MAX);
    array<int, 1> aB(size, B.begin(), B.end(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU {
        aB[idx] <<= aA[idx];
    });

	for (int i=0; i<size; ++i) { B[i] <<= A[i]; }	
	
    vector<int> C = aB;
    bool passed = Verify(C, B);

	return passed;
}

