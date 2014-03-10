// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test array_views without a data source can not be created in GPU context</summary>
//#Expects: Error: test.cpp\(20\) : error C3930
//#Expects: Error: test.cpp\(25\) : error C3930

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

// Test array_view of given extent	
void test() restrict(amp)
{
	array_view<int, 2> arrViewResult(16,16); // Results in compilation error
}

void test1() restrict(amp,cpu)
{
	array_view<int, 2> arrViewResult(16,16); // Results in compilation error
}


runall_result test_main()
{
    return runall_fail;
}
