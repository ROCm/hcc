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
/// <summary>Tests assigning a compatible element-type modifier (readonly gets read/write)</summary>

#include "../../../helper.h"
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<int> v_rw(20);
    Fill<int>(v_rw);
    array_view<int, 1> av_rw(20, v_rw);

    std::vector<int> v_ro(30);
    Fill<int>(v_ro);
    array_view<const int, 1> av_ro(30, v_ro);

    av_ro = av_rw;

    return verify_extent(av_ro, av_rw.get_extent()) && Verify(av_ro.data(), v_rw.data(), av_rw.get_extent().size()) ? runall_pass : runall_fail;
}
