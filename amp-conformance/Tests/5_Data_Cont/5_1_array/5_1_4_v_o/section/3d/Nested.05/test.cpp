// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test a nested section (at (1, 1, 0) of size (2, 3, 1) of a section (at (3, 3, 3) of size (5, 5, 5) of a 3D array(10, 10, 10) </summary>

#include <amptest/array_test.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    ArrayTest<float, 3> original(extent<3>(10, 10, 10));
    ArrayViewTest<float, 3> section1 = original.section(index<3>(3, 3, 3), extent<3>(5, 5, 5));
    ArrayViewTest<float, 3> section2 = section1.section(index<3>(1, 1, 0), extent<3>(2, 3, 1));

    // the index<1> parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<3>(3, 3, 3)) &&
        TestSection(original, section2, index<3>(4, 4, 3)) &&
        TestSection(section1, section2, index<3>(1, 1, 0))
        ? original.pass() : original.fail();
}
