// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a section (GPU) on a rank 3 array view using the convenience APIs</summary>

#include <amptest/array_test.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result scalar_values()
{
    ArrayTest<int, 3> original(extent<3>(2, 3, 4));

    original.set_value(index<3>(1, 2, 2), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 3> gpu_section = gpu_original.section(0, 1, 2, 2, 2, 2);
        gpu_section(0, 0, 0) = gpu_original(1, 2, 2);
    });

    ArrayViewTest<int, 3> section = original.section(original.arr().section(index<3>(0, 1, 2), extent<3>(2, 2, 2)), index<3>(0, 1, 2));
    section.set_known_value(index<3>(0, 0, 0), 13);

    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 13 && section.view()(1, 1, 0) == 13) ? original.pass() : original.fail();
}

runall_result only_index()
{
    ArrayTest<int, 3> original(extent<3>(2, 3, 4));

    original.set_value(index<3>(1, 2, 2), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 3> gpu_section = gpu_original.section(index<3>(0, 1, 2));
        gpu_section(0, 0, 0) = gpu_original(1, 2, 2);
    });

    ArrayViewTest<int, 3> section = original.section(original.arr().section(index<3>(0, 1, 2)), index<3>(0, 1, 2));
    section.set_known_value(index<3>(0, 0, 0), 13);

    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 13 && section.view()(1, 1, 0) == 13) ? original.pass() : original.fail();
}

runall_result only_extent()
{
    ArrayTest<int, 3> original(extent<3>(2, 3, 4));

    original.set_value(index<3>(1, 1, 1), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 3> gpu_section = gpu_original.section(extent<3>(2, 2, 2));
        gpu_section(0, 0, 0) = gpu_original(1, 1, 1);
    });

    ArrayViewTest<int, 3> section = original.section(original.arr().section(extent<3>(2, 2, 2)), index<3>(0, 0, 0));
    section.set_known_value(index<3>(0, 0, 0), 13);

    return (gpu_read(original.arr(),index<3>(0, 0, 0)) == 13 && section.view()(1, 1, 1) == 13) ? original.pass() : original.fail();
}

runall_result test_main()
{
    runall_result res;
	
	res &= REPORT_RESULT(scalar_values());
	res &= REPORT_RESULT(only_index());
	res &= REPORT_RESULT(only_extent());
	
	return res;
}

