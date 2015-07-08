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
/// <summary>View an AV as 3D (GPU)</summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    extent<3> ex(4, 5, 3);
    ArrayViewTest<int, 1> original(extent<1>(ex.size()));
    array_view<int, 1> original_av = original.view();

    extent_coordinate_nest<3> coordinates(ex);
    int linear = coordinates.get_linear(index<3>(2, 3, 1));
    parallel_for_each(extent<1>(1), [=](index<1>) __GPU {
        array_view<int, 3> rank3 = original_av.view_as(ex);
        // read and write between the original and higher rank view
        original_av[linear] = 17;
        rank3[index<3>(2, 3, 2)] = rank3[index<3>(2, 3, 1)];
    });

    original.set_known_value(index<1>(coordinates.get_linear(index<3>(2, 3, 1))), 17);
    original.set_known_value(index<1>(coordinates.get_linear(index<3>(2, 3, 2))), 17);


    return
        original.view()[coordinates.get_linear(index<3>(2, 3, 2))] == 17 &&
        original.view()[coordinates.get_linear(index<3>(2, 3, 1))] == 17
        ? original.pass() : original.fail();
}

