// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that operator[],operator()</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::Test;

using std::vector;

runall_result test1(accelerator_view &acc_view)
{
	runall_result result;
	extent<1> e(10);
	std::vector<int> src_v(e.size());
	for( int i = 0 ; i < static_cast<int>(e.size()) ; i++ )
	{
		src_v[i] = i+1;
	}

    array<int> src_arr(e, src_v.begin(),acc_view);
	vector<int> dst_v(e.size());
	array_view<int> dst_av(e, dst_v);

	// On CPU acceleators
	if( acc_view.get_accelerator().get_device_path() == accelerator::cpu_accelerator)
	{
		for( int i = 0 ; i < static_cast<int>(e.size()); i++)
		{
			if(src_arr[i] != i+1)
			{
				Log(LogType::Error, true) << "Data Mismatch found while testing Index operations on Array created on CPU accl, Expected:" << (i+1) << " Actual:" <<  src_arr[i] <<  std::endl;
				return runall_fail;
			}

			if(src_arr(i) != i+1)
			{
				Log(LogType::Error, true) << "Data Mismatch found during Index operations on Array created on CPU accl, Expected:" << (i+1) << " Actual:" <<  src_arr(i) <<  std::endl;
				return runall_fail;
			}
		}
		return result;
	}

	{

		dst_av.discard_data();

		parallel_for_each(src_arr.get_extent(),[&src_arr,dst_av](index<1> idx)restrict(amp){
			dst_av[idx] = src_arr[idx[0]];
		});
		dst_av.synchronize();
		result &= REPORT_RESULT(result &= Verify(dst_v, src_v));
	}

	{
		dst_av.discard_data();

		parallel_for_each(src_arr.get_extent(),[&src_arr,dst_av](index<1> idx)restrict(amp){
			dst_av[idx] = src_arr(idx[0]);
		});
		dst_av.synchronize();
		result &= REPORT_RESULT(result &= Verify(dst_v, src_v));
	}

	return result;
}


runall_result test2(accelerator_view &acc_view)
{
	runall_result result;
	int data[] = {
	1, 1 , 1, 1 , 1 ,
	2 , 2, 2, 2,  2 };

    extent<2> ext(2,5);
    array<int,2> src_arr(ext,data,acc_view);
	vector<int> expected_v(5);
	vector<int> actual_v(5);

	{
		// CPU
		std::fill(expected_v.begin(),expected_v.end(),1);
		array_view<int> dst_av = src_arr[0];
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<1>(ext[1]));

		// GPU
		dst_av.discard_data();
		parallel_for_each(extent<1>(1),[=,&src_arr](index<1>) restrict(amp,cpu){
			array_view<int> results = src_arr[0];
			for(int i = 0;i < static_cast<int>(dst_av.get_extent().size()) ;i++)
				dst_av[i] = results[i];
		});
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<1>(ext[1]));
	}

	{
		// CPU
		std::fill(expected_v.begin(),expected_v.end(),2);
		array_view<int> dst_av = src_arr(1);
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<1>(ext[1]));

		// GPU
		dst_av.discard_data();
		parallel_for_each(extent<1>(1),[=,&src_arr](index<1>) restrict(amp,cpu){
			array_view<int> results = src_arr(1);
			for(int i = 0;i < static_cast<int>(dst_av.get_extent().size()) ;i++)
				dst_av[i] = results[i];
		});
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<1>(ext[1]));
	}


	return result;
}

runall_result test3(accelerator_view &acc_view)
{
	runall_result result;
	int data[] = {
	1, 1 , 1, 1 , 1 ,
	2 , 2, 2, 2,  2 ,
	3, 3 , 3, 3 , 3 ,
	4 , 4, 4, 4,  4	,
	5 , 5 , 5 ,5 , 5,
	6 , 6 , 6 ,6 , 6 };

    extent<3> ext(2,5,3);
    array<int,3> src_arr(ext,data,acc_view);
	vector<int> expected_v(15);
	vector<int> actual_v(15);

	{
		// CPU
		std::fill(expected_v.begin(),expected_v.begin() + 5,1);
		std::fill(expected_v.begin()+5,expected_v.begin() + 10,2);
		std::fill(expected_v.begin()+10,expected_v.end(),3);
		array_view<int,2> dst_av = src_arr[0];
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<2>(ext[1],ext[2]));

		// GPU
		dst_av.discard_data();
		parallel_for_each(extent<1>(1),[=,&src_arr](index<1>) restrict(amp,cpu){
			array_view<int,2> results = src_arr[0];
			for(int i = 0;i < dst_av.get_extent()[0] ;i++)
			{
				for(int j = 0; j < dst_av.get_extent()[1];j++)
				{
					dst_av[i][j] = results[i][j];
				}
			}
		});
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<2>(ext[1],ext[2]));
	}

	{
		// CPU
		std::fill(expected_v.begin(),expected_v.begin() + 5,4);
		std::fill(expected_v.begin()+5,expected_v.begin() + 10,5);
		std::fill(expected_v.begin()+10,expected_v.end(),6);
		std::fill(actual_v.begin(),actual_v.end(),-1); // Initialising
		array_view<int,2> dst_av = src_arr(1);
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<2>(ext[1],ext[2]));

		// GPU
		dst_av.discard_data();
		parallel_for_each(extent<1>(1),[=,&src_arr](index<1>) restrict(amp,cpu){
			array_view<int,2> results = src_arr(1);
			for(int i = 0;i < dst_av.get_extent()[0] ;i++)
			{
				for(int j = 0; j < dst_av.get_extent()[1];j++)
				{
					dst_av[i][j] = results[i][j];
				}
			}
		});
		copy(dst_av,actual_v.begin());
		result &= REPORT_RESULT(Verify(actual_v, expected_v));
		result &= REPORT_RESULT(dst_av.get_extent() == extent<2>(ext[1],ext[2]));
	}

	return result;
}


runall_result test_main()
{
    accelerator_view acc_view = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    runall_result result;
    accelerator_view cpu_view = accelerator(accelerator::cpu_accelerator).get_default_view();

    result &= REPORT_RESULT(test1(acc_view));
    result &= REPORT_RESULT(test1(cpu_view));
	result &= REPORT_RESULT(test2(acc_view));
	result &= REPORT_RESULT(test3(acc_view));
    return result;
}
