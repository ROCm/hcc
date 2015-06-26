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
/// <summary>Test a nested section (0, 4) of a section (4, 4) of a 1D array(10) </summary>

#include <amptest/array_view_test.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<long, 1> original(extent<1>(10));
    ArrayViewTest<long, 1> section1 = original.section(original.view().section(4, 4), index<1>(4));
    ArrayViewTest<long, 1> section2 = section1.section(section1.view().section(0, 4), index<1>(0));

    // the index<1> parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<1>(4)) &&
        TestSection(original, section2, index<1>(4)) &&
        TestSection(section1, section2, index<1>(0))
        ? original.pass() : original.fail();
}
