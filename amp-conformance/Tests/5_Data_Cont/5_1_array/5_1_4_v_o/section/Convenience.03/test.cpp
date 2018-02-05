// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a section of a rank 3 array  using the convenience APIs</summary>

#include <amptest/array_test.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result scalar_values()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(0, 1, 2, 2, 2, 2), index<3>(0, 1, 2));

    original.set_value(index<3>(1, 2, 2), 13);

    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);

    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 15 && section.view()(1, 1, 0) == 13) ? original.pass() : original.fail();
}

runall_result only_index()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(index<3>(0, 1, 2)), index<3>(0, 1, 2));

    original.set_value(index<3>(1, 2, 2), 13);

    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);

    return (gpu_read(original.arr(),index<3>(0, 1, 2)) == 15 && section.view()(1, 1, 0) == 13) ? original.pass() : original.fail();
}

runall_result only_extent()
{
	ArrayTest<int, 3> original(extent<3>(2, 3, 4));
    ArrayViewTest<int, 3> section = original.section(original.arr().section(extent<3>(2, 2, 2)), index<3>(0, 0, 0));

    original.set_value(index<3>(1, 1, 1), 13);

    section.view()(0, 0, 0) = 15;
    section.set_known_value(index<3>(0, 0, 0), 15);
	
    return (gpu_read(original.arr(),index<3>(0, 0, 0)) == 15 && section.view()(1, 1, 1) == 13)? original.pass() : original.fail();
}

runall_result test_main()
{
    runall_result res;
	
	res &= REPORT_RESULT(scalar_values());
	res &= REPORT_RESULT(only_index());
	res &= REPORT_RESULT(only_extent());
	
	return res;
}

