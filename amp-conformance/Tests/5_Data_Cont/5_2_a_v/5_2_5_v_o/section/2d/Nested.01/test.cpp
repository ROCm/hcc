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
/// <summary>Test a (left-edge) nested section (at (0, 0) of size (2, 2) of a section (at (0, 0) of size (2, 4) of a 2D array(10, 10) </summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<float, 2> original(extent<2>(10, 10));
    ArrayViewTest<float, 2> section1 = original.section(original.view().section(index<2>(0, 0), extent<2>(2, 4)), index<2>(0, 0));
    ArrayViewTest<float, 2> section2 = section1.section(section1.view().section(index<2>(0, 0), extent<2>(2, 2)), index<2>(0, 0));

    // the index<1> parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<2>(0, 0)) &&
        TestSection(original, section2, index<2>(0, 0)) &&
        TestSection(section1, section2, index<2>(0, 0))
        ? original.pass() : original.fail();
}
