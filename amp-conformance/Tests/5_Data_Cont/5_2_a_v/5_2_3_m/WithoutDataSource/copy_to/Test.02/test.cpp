// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Test 'copy_to' member function on array_views without a data source</summary>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array_view without data source , before p_f_e
* Destination => array object
*/
runall_result test1(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	array_view<int,2> arrViewSrc(M,N);
	array<int,2> arrDest(M,N);

	arrViewSrc.copy_to(arrDest); // Copying to Array
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,arrDest));
	return result;
}

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array_view without data source , before p_f_e
* Destination => array_view with data source
*/
runall_result test2(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	array_view<int,2> arrViewSrc(M,N);
	std::vector<int> destVect( M * N , -1 );
	array_view<int,2> arrViewDest(M,N,destVect);

	arrViewSrc.copy_to(arrViewDest); // Copying to Array
	arrViewDest.synchronize();
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,arrViewDest));
	return result;
}
runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

	res &= REPORT_RESULT(test1(av));
	res &= REPORT_RESULT(test2(av));
    return res;
}

