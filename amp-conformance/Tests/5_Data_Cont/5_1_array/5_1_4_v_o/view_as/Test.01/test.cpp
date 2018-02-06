// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>View an 1-D Array as 3D and do the updations in CPU</summary>

#include <amptest/array_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>
#include <vector>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    extent<3> ex(4, 5, 3);
    ArrayTest<int, 1> original(extent<1>(ex.size()));

    extent_coordinate_nest<3> coordinates(ex);
    array_view<int, 3> rank3 = original.arr().view_as(ex);

    // read and write between the original and higher rank view
	index<3> set_original(2, 3, 1);
	index<1> set_original_linear(coordinates.get_linear(set_original));
	gpu_write<int,1>(original.arr(), set_original_linear, 17);
    original.set_known_value(set_original_linear, 17);

	index<3> set_viewAs(2,3,2);
	index<1> set_viewAs_linear(coordinates.get_linear(set_viewAs));
    rank3[set_viewAs] = rank3[set_original];
    original.set_known_value(set_viewAs_linear, 17);

    return
        gpu_read<int,1>(original.arr(), set_viewAs_linear) == 17 &&
        rank3[set_original] == 17
        ? original.pass() : original.fail();
}

