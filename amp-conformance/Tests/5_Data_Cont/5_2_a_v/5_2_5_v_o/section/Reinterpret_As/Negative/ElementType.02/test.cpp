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
/// <summary>Attempt to call reinterpret_as with a GPU prohibited type on the GPU</summary>
//#Expects: Error: error C2338
//#Expects: Error: test.cpp\(36\)

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<int> v(10);
    array_view<int, 1> av(v.size(), v);
    parallel_for_each(av.get_extent(), [=](index<1> i) __GPU {
        array_view<short, 1> r = av.reinterpret_as<short>();
    });

    // this test should not compile
    return runall_fail;
}

