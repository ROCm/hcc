// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>View an 1D Array as a shorter 1D AV</summary>

#include <amptest/array_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>
#include <vector>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    ArrayTest<int, 1> original(extent<1>(15));
    array<int, 1> &original_arr = original.arr();
    array_view<int, 1> shorter = original_arr.view_as(extent<1>(10));

    // read and write between the original and higher rank view
    index<1> set_original(9);
	gpu_write<int,1>(original_arr, set_original, 17);
    original.set_known_value(set_original, 17);

	index<1> set_view(8);
    shorter[set_view] = shorter[9];
    original.set_known_value(set_view, 17);

    return
        gpu_read<int,1>(original_arr, set_view) == 17 &&
        shorter[set_original] == 17
        ? original.pass() : original.fail();
}

