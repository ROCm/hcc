// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test that array_views without a data source results in compilation error, when Rank and the number of dimensions of array_view extent mismatch</summary>
//#Expects: Error: error C2664

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
	
void test()
{
	array_view<int,1> arr1(10,10);
	
	array_view<int,2> arr2_1(10);
	array_view<int,2> arr2_2(10,10,10);
	
	array_view<int,3> arr3_1(10);
	array_view<int,3> arr3_2(10,10);
}

runall_result test_main()
{
    return runall_fail;
}
