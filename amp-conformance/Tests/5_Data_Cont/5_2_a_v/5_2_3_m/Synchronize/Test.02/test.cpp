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
/// <summary>Modify array_view data in a parallel for each, and then use synchronize_async to see the updates</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    accelerator::set_default(require_device(device_flags::NOT_SPECIFIED).get_device_path());

    int size = 30;
    std::vector<int> v(size);
    Fill<int>(v);
    array_view<int, 1> av(size, v);

    parallel_for_each(av.get_extent(), [av](index<1> i) __GPU {
        av[i] = 3;
    });

    std::shared_future<void> w = av.synchronize_async();
    w.wait();

    // All elements should equal 3
    return (std::count(v.begin(), v.end(), 3) == size);
}

