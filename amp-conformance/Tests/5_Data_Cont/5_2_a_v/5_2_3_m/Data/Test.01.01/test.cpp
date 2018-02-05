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
/// <summary>An array view can have it's data pointer accessed in the GPU context.
/// This time the backing store is a staging buffer</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    accelerator cpu(accelerator::cpu_accelerator);
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view acc_view = device.get_default_view();

    const int size = 20;
    vector<int> vec(size);
    Fill<int>(vec.data(), size);

    extent<1> ex(size);
    array<int, 1> arr(ex, vec.begin(), cpu.get_default_view(), acc_view);

    // access this on the GPU
    array_view<const int, 1> av(arr);

    // fill this on the GPU
    vector<int> result_v(size);
    array_view<int, 1> result(20, result_v);

    parallel_for_each(av.get_extent(), [av, result](index<1> i) __GPU {
       result.data()[i[0]] =  av.data()[i[0]]; // get and set data using the pointer
    });

    return Verify(result.data(), av.data(), size) ? runall_pass : runall_fail;
}

