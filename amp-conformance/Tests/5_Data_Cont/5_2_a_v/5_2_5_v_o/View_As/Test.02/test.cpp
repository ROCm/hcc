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
/// <summary>View a const AV as 2D (CPU)</summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    extent<2> ex(4, 5);
    extent_coordinate_nest<2> coordinates(ex);
    ArrayViewTest<const int, 1> original(extent<1>(ex.size()));

    // set a value in the underlying data
    original.set_value(index<1>(coordinates.get_linear(index<2>(2, 3))), 17);

    array_view<const int, 2> rank2 = original.view().view_as(ex);

    original.set_value(index<1>(coordinates.get_linear(index<2>(2, 2))), 13);

    return
        rank2[index<2>(2, 3)] == 17 &&
        rank2[index<2>(2, 2)] == 13
        ? original.pass() : original.fail();
}

