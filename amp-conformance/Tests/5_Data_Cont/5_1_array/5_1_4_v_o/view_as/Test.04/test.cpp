// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Attempt to call view_as on array object as another array_view object of same rank</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <amptest/array_test.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    std::vector<int> v(10);
	std::fill(v.begin(),v.begin()+5,-1);
	std::fill(v.begin()+5,v.end(),-2);
    array<int, 2> arr(5, 2, v.begin());
	array_view<int, 2> r = arr.view_as(extent<2>(2, 5));
	
	index<2> set_original(2,1); // Set 6th element in the vector ie., 3rd row, 2nd element
	index<2> read_view(1,0);  // 6th element in the 'view_as' AV is 2nd row, 1st element
	
	index<2> set_view(1,4);  // setting 10th element in the view_as AV ie., 2nd row, 5th element
	index<2> read_original(4,1); // 10th element in the original array is 5th row , 2nd element.
	
	gpu_write(arr,set_original,0);
	r[set_view] = 0;
	
	return ( (gpu_read(arr,read_original) == 0)  && (r[read_view] == 0));
}

