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
* Source => array_view without data source , passed to p_f_e
* Destination => array object
*/
runall_result test1(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	array_view<int,2> arrViewSrc(M,N);
	array<int,2> arrDest(M,N);

	parallel_for_each(av, arrViewSrc.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSrc[idx] = idx[0] * 10 + idx[1];
	});

	arrViewSrc.copy_to(arrDest); // Copying to Array
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,arrDest));
	return result;
}

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array_view without data source , passed to p_f_e
* Destination => another array_view object having data source
*/
runall_result test2(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	array_view<int,2> arrViewSrc(M,N);

	parallel_for_each(av, arrViewSrc.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSrc[idx] = idx[0] * 10 + idx[1];
	});

	std::vector<int> destVect( M * N , 0 );
	array_view<int,2> destArrView(M,N,destVect);
	arrViewSrc.copy_to(destArrView); // Copying to Array
	destArrView.synchronize();
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,destArrView));
	return result;
}

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array_view without data source , passed to p_f_e
* Destination => another array_view object having no data source
*/
runall_result test3(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	array_view<int,2> arrViewSrc(M,N);

	parallel_for_each(av, arrViewSrc.get_extent(), [=](const index<2> &idx) restrict(amp) {
		arrViewSrc[idx] = idx[0] * 10 + idx[1];
	});

	array_view<int,2> destArrView(M,N);
	arrViewSrc.copy_to(destArrView); // Copying to Arrayview having no data source
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,destArrView));
	//destArrView.synchronize();

	return result;
}

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array object
* Destination => array_view object having no data source
*/
runall_result test4(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	std::vector<int> vecSrc(M * N);
	std::generate(vecSrc.begin(), vecSrc.end(), rand);
	array<int,2> arrSrc(M,N,vecSrc.begin(), av);
	array_view<int,2> arrViewDest(M,N);

	arrSrc.copy_to(arrViewDest); // Copying to Arrayview having no data source
	arrViewDest.synchronize();
	result &= REPORT_RESULT(VerifyDataOnCpu(arrSrc,arrViewDest));

	// Verifying Data on Gpu
	int comp_result = 0;
	array_view<int,1> av_compare_result(1,&comp_result);
	parallel_for_each(av,arrSrc.get_extent(),[=,&arrSrc](index<2> idx) restrict(amp){
		if(arrSrc(idx) != arrViewDest(idx))
		{
			av_compare_result[0] = 1;
		}
	});
	av_compare_result.synchronize();
	result &= REPORT_RESULT(comp_result == 0);
	return result;
}

/*
* Testing 'copy_to' on array_view object created without data source.
* Source => array_view object having data source
* Destination => array_view object having no data source
*/
runall_result test5(const accelerator_view &av)
{
	runall_result result;
	const int M = 256;
	const int N = 256;

	std::vector<int> vecSrc(M * N);
	std::generate(vecSrc.begin(), vecSrc.end(), rand);
	array_view<int,2> arrViewSrc(M,N,vecSrc);
	array_view<int,2> arrViewDest(M,N);

	arrViewSrc.copy_to(arrViewDest); // Copying to Arrayview having no data source
	result &= REPORT_RESULT(VerifyDataOnCpu(arrViewSrc,arrViewDest));

	int comp_result = 0;
	array_view<int,1> av_compare_result(1,&comp_result);
	parallel_for_each(av,arrViewSrc.get_extent(),[=](index<2> idx) restrict(amp){
		if(arrViewSrc(idx) != arrViewDest(idx))
		{
			av_compare_result[0] = 1;
		}
	});
	av_compare_result.synchronize();
	result &= REPORT_RESULT( comp_result == 0 );

	return result;
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
    runall_result res;

	res &= REPORT_RESULT(test1(av));
	res &= REPORT_RESULT(test2(av));
	res &= REPORT_RESULT(test3(av));
	res &= REPORT_RESULT(test4(av));
	res &= REPORT_RESULT(test5(av));
    return res;
}

