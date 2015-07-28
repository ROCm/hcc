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
/// <summary>Test that an array_view can be self-assigned a section of itself in a cpu restricted function</summary>

#include "../../../helper.h"
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    // create two AVs with different data
    std::vector<float> v1(20);
    Fill<float>(v1);
    array_view<float, 3> av1(5, 2, 2, v1);

    array_view<float, 3> section = av1.section(index<3>(1, 1, 1), extent<3>(3, 1, 1));

    // self-assign a section
    av1 = av1.section(index<3>(1, 1, 1), extent<3>(3, 1, 1));

    return verify_extent(av1, section.get_extent()) && VerifyDataOnCpu(av1, section) ? runall_pass : runall_fail;
}
