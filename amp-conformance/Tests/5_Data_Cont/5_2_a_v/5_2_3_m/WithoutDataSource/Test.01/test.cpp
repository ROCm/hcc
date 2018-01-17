// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test that member functions on array_views without a data source, does nt throw exceptions or given errors, when used in cpu context</summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

const int M = 256;
const int N = 256;

bool test1()
{		
	array_view<int,2> arrViewSrc(M,N);
	array_view<int,2> arrDest(M,N);
	
	// Verifying that the below operation should not throw any exception.
	arrViewSrc.copy_to(arrDest); // Copying to Array view with out data source
	
	return true;
}

bool test2()
{
	array_view<int,2> arrViewSrc(M,N);
	array_view<int,2> arrDest(M,N);
	
	// Verifying that the below operation should not throw any exception.
	arrDest = arrViewSrc; // Copying to Array view with out data source
	
	return true;
}

bool test3()
{
	array_view<int,2> arrViewSrc(M,N);
	
	// Verifying that the below operation should not throw any exception.
	arrViewSrc(1,1);
	
	return true;
}

bool test4()
{
	array_view<int,1> arrViewSrc(M);
	
	// Verifying that the below operation should not throw any exception.
	arrViewSrc[1];
	
	return true;
}

runall_result test_main()
{
    runall_result res;
	
    res &= REPORT_RESULT(test1());
    res &= REPORT_RESULT(test2());
    res &= REPORT_RESULT(test3());
    res &= REPORT_RESULT(test4());

    return res;
}
