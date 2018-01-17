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
/// <summary>Reinterpret an AV of 3 floats as float (GPU)</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    class Foo
    {
    public:
        float r;
        float b;
        float g;
    };

    std::vector<float> v(10 * 3);
    Fill(v);

    array_view<Foo, 1> av_rbg(10, reinterpret_cast<Foo *>(v.data()));

    int expected_size = av_rbg.reinterpret_as<float>().get_extent().size();

    // reinterpret on the GPU and copy back
    std::vector<float> results_v(expected_size);
    array_view<float, 1> results(static_cast<int>(results_v.size()), results_v);
    parallel_for_each(av_rbg.reinterpret_as<float>().get_extent(), [=](index<1> i) __GPU {
        array_view<const float, 1> av_float = av_rbg.reinterpret_as<float>();
        results[i] = av_float[i];
    });

    return Verify<float>(reinterpret_cast<float *>(av_rbg.data()), results.data(), expected_size) ? runall_pass : runall_fail;
}
