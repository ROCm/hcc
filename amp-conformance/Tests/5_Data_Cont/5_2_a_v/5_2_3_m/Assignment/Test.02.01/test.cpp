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
/// <summary>Test that an array_view can be self-assigned in a GPU restricted function</summary>

#include "../../../helper.h"
#include <amptest.h>
#include <numeric>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view acc_view = device.get_default_view();

    // now assign av1 on the GPU and copy back the data
    std::vector<int> results_v(10, -1);
	array_view<int, 2> results(5, 2, results_v);

	std::vector<int> expected(results_v.size());
	std::iota(expected.begin(), expected.end(), 0);

    // something here will break if the extent aren't copied properly
    parallel_for_each(results.get_extent(), [=](index<2> i) __GPU {
		int data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		array_view<int, 2> av1(5, 2, data);
        av1 = av1;
        results[i] = av1[i];
    });
    results.synchronize(); // push pending writes to the local vector

    return Verify(results_v, expected) ? runall_pass : runall_fail;
}
