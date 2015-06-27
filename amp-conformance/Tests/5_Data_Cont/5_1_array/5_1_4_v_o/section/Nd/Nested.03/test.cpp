// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test a nested section of a section of a 6D array(4, 4, 4, 4, 4, 4) </summary>

#include <amptest/array_test.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    int original_ex[6] = {4, 4, 4, 4, 4, 4};

    int section1_offset[6] = { 2, 0, 0, 0, 2, 2 };
    int section1_ex[6] =     { 2, 4, 4, 1, 2, 2 };

    int section2_offset[6] = { 0, 1, 3, 0, 0, 0 };
    int section2_ex[6] =     { 2, 2, 1, 1, 2, 2 };

    ArrayTest<float, 6> original((extent<6>(original_ex)));
    ArrayViewTest<float, 6> section1 = original.section(index<6>(section1_offset), extent<6>(section1_ex));
    ArrayViewTest<float, 6> section2 = section1.section(index<6>(section2_offset), extent<6>(section2_ex));

    // the index parameters here are of the offset (second - first)
    return
        TestSection(original, section1, index<6>(section1_offset)) &&
        TestSection(original, section2, index<6>(section1_offset) + index<6>(section2_offset)) &&
        TestSection(section1, section2, index<6>(section2_offset))
        ? original.pass() : original.fail();
}
