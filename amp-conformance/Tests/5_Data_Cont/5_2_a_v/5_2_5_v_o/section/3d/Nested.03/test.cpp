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
/// <summary>Test a nested section (at (0, 0, 2) of size (1, 1, 5) of a section (at (2, 2, 0) of size (1, 1, 10) of a 3D array(10, 10, 10) </summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<float, 3> original(extent<3>(10, 10, 10));
    ArrayViewTest<float, 3> section1 = original.section(index<3>(2, 2, 0), extent<3>(1, 1, 10));
    ArrayViewTest<float, 3> section2 = section1.section(index<3>(0, 0, 2), extent<3>(1, 1, 5));

    // the index parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<3>(2, 2, 0)) &&
        TestSection(original, section2, index<3>(2, 2, 2)) &&
        TestSection(section1, section2, index<3>(0, 0, 2))
        ? original.pass() : original.fail();
}
