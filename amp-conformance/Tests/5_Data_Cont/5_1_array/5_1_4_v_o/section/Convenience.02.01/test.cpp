// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a section (GPU) on a rank 2 array view using the convenience APIs on GPU</summary>

#include <amptest/array_test.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result scalar_values()
{
    ArrayTest<int, 2> original(extent<2>(10, 10));

    original.set_value(index<2>(3, 4), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 2> gpu_section = gpu_original.section(2, 3, 5, 2);
        gpu_section(3, 1) = gpu_original(3, 4);
    });

    ArrayViewTest<int, 2> section = original.section(original.arr().section(2, 3, 5, 2), index<2>(2, 3));
    section.set_known_value(index<2>(3, 1), 13);

    return (gpu_read(original.arr(),index<2>(5, 4)) == 13 && section.view()(1, 1) == 13) ? original.pass() : original.fail();
}

runall_result only_index()
{
    ArrayTest<int, 2> original(extent<2>(10, 10));

    original.set_value(index<2>(3, 4), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 2> gpu_section = gpu_original.section(index<2>(2, 3));
        gpu_section(3, 1) = gpu_original(3, 4);
    });

    ArrayViewTest<int, 2> section = original.section(original.arr().section(index<2>(2, 3)), index<2>(2, 3));
    section.set_known_value(index<2>(3, 1), 13);

    return (gpu_read(original.arr(),index<2>(5, 4)) == 13 && section.view()(1, 1) == 13) ? original.pass() : original.fail();
}

runall_result only_extent()
{
    ArrayTest<int, 2> original(extent<2>(10, 10));

    original.set_value(index<2>(3, 4), 13);

    // create a section on the GPU, use it to read and write
    auto &gpu_original = original.arr();
    parallel_for_each(extent<1>(1), [=,&gpu_original](index<1>) __GPU {
        array_view<int, 2> gpu_section = gpu_original.section(extent<2>(5, 5));
        gpu_section(3, 1) = gpu_original(3, 4);
    });

    ArrayViewTest<int, 2> section = original.section(original.arr().section(extent<2>(5, 5)), index<2>(0, 0));
    section.set_known_value(index<2>(3, 1), 13);

    return (gpu_read(original.arr(),index<2>(3, 1)) == 13 && section.view()(3, 4) == 13) ? original.pass() : original.fail();
}

runall_result test_main()
{
    runall_result res;
	
	res &= REPORT_RESULT(scalar_values());
	res &= REPORT_RESULT(only_index());
	res &= REPORT_RESULT(only_extent());
	
	return res;
}
