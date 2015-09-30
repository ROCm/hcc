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
/// <summary>Retrieve a projection of a section</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

#include <iostream>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<int, 2> original(extent<2>(5, 5));

    // this section is from [2-5), 2-5) of the original
    ArrayViewTest<int, 2> section = original.section(original.view().section(index<2>(2, 2)), index<2>(2, 2));

    // create a projection -- this is row 2 of the original (2, [2-5))
    ArrayViewTest<int, 1, 2> projection = section.projection(section.view()[0], 0);

    // set some data in the original
    original.view()[index<2>(2, 3)] = 13;
    original.set_known_value(index<2>(2, 3), 13);

    // set some data in the section -- (2, 4) in the original
    section.view()[index<2>(0, 2)] = 17;
    section.set_known_value(index<2>(0, 2), 17);

    // set some data in the projection -- (2, 2) in the original
    projection.view()[index<1>(0)] = 11;
    projection.set_known_value(index<1>(0), 11);

    // verify each data point through the array_view interface
    return
        original.view()(2, 2) == 11 &&
        original.view()(2, 3) == 13 &&
        original.view()(2, 4) == 17 &&
        section.view()(0, 0) == 11 &&
        section.view()(0, 1) == 13 &&
        section.view()(0, 2) == 17 &&
        projection.view()(0) == 11 &&
        projection.view()(1) == 13 &&
        projection.view()(2) == 17
        ? original.pass() : original.fail();
}

