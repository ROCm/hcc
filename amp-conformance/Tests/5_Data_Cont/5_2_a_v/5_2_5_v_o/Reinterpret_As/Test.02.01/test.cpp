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
/// <summary>Reinterpret an AV of const float as const int (GPU)</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<float> v(10);
    Fill(v);

    array_view<const float, 1> av_float(static_cast<int>(v.size()), v);

    // reinterpret on the GPU and copy back
    std::vector<int> results_v(v.size());
    array_view<int, 1> results(static_cast<int>(results_v.size()), results_v);
    parallel_for_each(av_float.get_extent(), [=](index<1> i) __GPU {
        array_view<const int, 1> av_int = av_float.reinterpret_as<int>();
        results[i] = av_int[i];
    });

    return Verify<int>(reinterpret_cast<const int *>(av_float.data()), results.data(), v.size()) ? runall_pass : runall_fail;
}

