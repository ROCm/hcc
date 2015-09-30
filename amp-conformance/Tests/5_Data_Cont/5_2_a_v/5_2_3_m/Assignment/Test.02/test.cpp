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
/// <summary>Test that an array_view can be self-assigned in a cpu restricted function</summary>

#include "../../../helper.h"
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    // create two AVs with different data
    std::vector<int> v1(20);
    Fill<int>(v1);
    array_view<int, 3> av1(5, 2, 2, v1);

    // self-assign
    av1 = av1;
    return verify_extent(av1, av1.get_extent()) && VerifyDataOnCpu(av1, av1) ? runall_pass : runall_fail;
}
