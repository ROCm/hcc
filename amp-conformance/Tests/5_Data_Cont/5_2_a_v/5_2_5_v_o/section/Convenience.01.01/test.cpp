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
/// <summary>Create a section or a rank 1 array view using the convenience APIs on GPU</summary>

#include <amptest/array_view_test.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result scalar_values()
{
    ArrayViewTest<int, 1> original(extent<1>(10));
    ArrayViewTest<int, 1> section = original.section(original.view().section(2, 5), index<1>(2));

	original.view()(2) = 13;
    original.set_known_value(index<1>(2), 13);

    // create a section on the GPU, use it to read and write
    auto gpu_original = original.view();
    parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp) {
        array_view<int, 1> gpu_section = gpu_original.section(2, 5);
        gpu_section(3) = gpu_original(2);
    });

    section.set_known_value(index<1>(3), 13);

    return (original.view()(5) == 13 && section.view()(0) == 13) ? original.pass() : original.fail();
}

runall_result only_index()
{
    ArrayViewTest<int, 1> original(extent<1>(10));
    ArrayViewTest<int, 1> section = original.section(original.view().section(index<1>(2)), index<1>(2));

	original.view()(2) = 13;
    original.set_known_value(index<1>(2), 13);

    // create a section on the GPU, use it to read and write
    auto gpu_original = original.view();
    parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp) {
        array_view<int, 1> gpu_section = gpu_original.section(index<1>(2));
        gpu_section(3) = gpu_original(2);
    });

    section.set_known_value(index<1>(3), 13);

    return (original.view()(5) == 13 && section.view()(0) == 13) ? original.pass() : original.fail();
}

runall_result only_extent()
{
    ArrayViewTest<int, 1> original(extent<1>(10));
    ArrayViewTest<int, 1> section = original.section(original.view().section(extent<1>(5)), index<1>(0));

	original.view()(2) = 13;
    original.set_known_value(index<1>(2), 13);

    // create a section on the GPU, use it to read and write
    auto gpu_original = original.view();
    parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp) {
        array_view<int, 1> gpu_section = gpu_original.section(extent<1>(5));
        gpu_section(1) = gpu_original(2);
    });

    section.set_known_value(index<1>(1), 13);

    return (original.view()(1) == 13 && section.view()(2) == 13) ? original.pass() : original.fail();
}

runall_result test_main()
{
    runall_result res;
	
	res &= REPORT_RESULT(scalar_values());
	res &= REPORT_RESULT(only_index());
	res &= REPORT_RESULT(only_extent());
	
	return res;
}

