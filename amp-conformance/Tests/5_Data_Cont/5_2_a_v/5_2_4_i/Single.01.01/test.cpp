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
/// <summary>Use a single integer index to retrieve edge-values of an Array View<1></summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    auto t = ArrayViewTest<int, 1>::sequential<0>(extent<1>(10));
    array_view<int, 1> av = t.view();

    ArrayViewTest<int, 1> results(t.view().get_extent());
    array_view<int, 1> results_v = results.view();

    // access values on the GPU
    parallel_for_each(extent<1>(1), [results_v, av] (index<1>) __GPU {
        results_v[0] = av[0];
        results_v[9] = av[9];
    });
    results.set_known_value(index<1>(0), 0);
    results.set_known_value(index<1>(9), 9);

    return results.view()[0] == 0 && results.view()[9] == 9 ? t.pass() : t.fail();
}

