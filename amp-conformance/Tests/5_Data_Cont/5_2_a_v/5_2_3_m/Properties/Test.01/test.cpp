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
/// <summary>Access all properties using the get_foo() style</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

runall_result test_main()
{
    const int size = 10 * 10 * 1;

    vector<int> vec(size);
    Fill<int>(vec.data(), size);

    accelerator_view acc_view = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    extent<3> ex(10, 10, 1);
    array<int, 3> arr(ex, vec.begin(), acc_view);

    array_view<int, 3> av(arr);

    if(arr.get_extent() != av.get_extent()) // verify extent
    {
        printf("array and array_view extents do not match. FAIL!\n");
        return runall_fail;
    }

    return runall_pass;
}

