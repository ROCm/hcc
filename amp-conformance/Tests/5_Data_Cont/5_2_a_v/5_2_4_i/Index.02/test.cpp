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
/// <summary>Use an index<N> to retrieve values from a section of an array_view</summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<int, 2> original(extent<2>(3, 3));
    ArrayViewTest<int, 2> section = original.section(original.view().section(index<2>(1, 1)), index<2>(1, 1));

    // set a value through the section -- this is (1,1) in the original view
    section.view()[index<2>(0, 0)] = 5;
    section.set_known_value(index<2>(0, 0), 5);

    // set a value through the original -- this is (1, 0) in the section
    original.view()[index<2>(2, 1)] = 2;
    original.set_known_value(index<2>(2, 1), 2);

    return
        original.view()[index<2>(1, 1)] == 5 &&
        section.view()[index<2>(1, 0)] == 2
        ? original.pass() : original.fail();
}

