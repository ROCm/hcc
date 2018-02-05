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
/// <summary>Use a single integer to retrieve projections of an Array View (higher than rank 1) (GPU)</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<int, 3> original(extent<3>(2, 3, 4));

    original.view()[index<3>(0, 1, 2)] = 13;
    original.view()[index<3>(1, 2, 2)] = 17;
    original.set_known_value(index<3>(0, 1, 2), 13);
    original.set_known_value(index<3>(1, 2, 2), 17);

    // create projections and set data on the GPU
    array_view<int, 3> original_av = original.view();
    parallel_for_each(extent<1>(1), [original_av](index<1>) __GPU {
        array_view<int, 2> gpu_proj0 = original_av[0];
        array_view<int, 2> gpu_proj1 = original_av[1];

        gpu_proj0[index<2>(1, 1)] = 11;
        gpu_proj1[index<2>(1, 1)] = 12;
    });

    // create tracking structures for the projections
    ArrayViewTest<int, 2, 3> proj0 = original.projection(original.view()[0], 0);
    ArrayViewTest<int, 2, 3> proj1 = original.projection(original.view()[1], 1);
    proj0.set_known_value(index<2>(1, 1), 11);
    proj1.set_known_value(index<2>(1, 1), 12);

    // verify each data point through the array_view interface
    return
        original.view()(0, 1, 1) == 11 &&
        original.view()(1, 1, 1) == 12 &&
        proj0.view()(1, 2) == 13 &&
        proj1.view()(2, 2) == 17
        ? original.pass() : original.fail();
}

