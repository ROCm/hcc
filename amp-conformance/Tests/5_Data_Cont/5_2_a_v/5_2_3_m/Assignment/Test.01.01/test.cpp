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
/// <summary>Test that an array_view can be assigned from an array_view in a GPU restricted function</summary>

#include "../../../helper.h"
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view acc_view = device.get_default_view();

    // create two AVs with different data
    std::vector<int> v1(20);
    Fill<int>(v1);
    array_view<int, 2> av1(5, 4, v1);

    // now assign av2 on the GPU and copy back the data
    std::vector<int> results_v(v1.size());
    array_view<int, 2> results(5, 4, results_v);

    // something here will break if the extent aren't copied properly
    parallel_for_each(av1.get_extent(), [=](index<2> i) __GPU {
        // av2 should now have the same extent and data as av1
		int av2_data[30] = {0};
		array_view<int, 2> av2(3, 10, av2_data);
        av2 = av1;
        results[i] = av2[i];
    });
    results.synchronize(); // push pending writes to the local vector

    return Verify(results_v, v1) ? runall_pass : runall_fail;
}
