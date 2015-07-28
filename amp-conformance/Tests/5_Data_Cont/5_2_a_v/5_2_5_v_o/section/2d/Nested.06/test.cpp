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
/// <summary>Test a completely nested section (at (1, 1) of size (2, 2) of a section (at (3, 3) of size (4, 4) of a 2D array(10, 10) </summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<float, 2> original(extent<2>(10, 10));
    ArrayViewTest<float, 2> section1 = original.section(original.view().section(index<2>(3, 3), extent<2>(4, 4)), index<2>(3, 3));
    ArrayViewTest<float, 2> section2 = section1.section(section1.view().section(index<2>(1, 1), extent<2>(2, 2)), index<2>(1, 1));

    // the index<1> parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<2>(3, 3)) &&
        TestSection(original, section2, index<2>(4, 4)) &&
        TestSection(section1, section2, index<2>(1, 1))
        ? original.pass() : original.fail();
}
