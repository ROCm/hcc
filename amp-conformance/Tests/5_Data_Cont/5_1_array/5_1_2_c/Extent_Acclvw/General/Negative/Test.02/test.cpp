// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator_view specialized construtors - same device different views.</summary>

#include "./../../../../constructor.h"
#include <amptest_main.h>

int iteration = 0;
int rank_step = 0;

template<typename _type, int _rank>
bool test_feature(const vector<accelerator>& devices)
{
	int edata[_rank];
	for (int i = 0; i < _rank; i++)
		edata[i] = 3;
	
	printf("Found %d devices\n", devices.size());

	for (size_t i = 0, iteration = 0; i < devices.size()-1; i++, iteration++)
	{
		accelerator device1 = devices[i];
		accelerator device2 = devices[i];

		if (_rank > 0)
		{
			const int rank = 1;
			array<_type, rank> src1(edata[0], device1.create_view(queuing_mode_immediate));
			array<_type, rank> src2(edata[0], device2.create_view(queuing_mode_automatic));

			// let the kernel initialize data;
			extent<1> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<_type> opt1(e1.size());
			opt1 = src1;
			vector<_type> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return false;
			}
			printf ("Passed for rank %d\n", rank);
			rank_step = rank;
		}
		if (_rank > 1)
		{
			const int rank = 2;
			rank_step = rank;
			array<_type, rank> src1(edata[0], edata[1], device1.create_view(queuing_mode_immediate));
			array<_type, rank> src2(edata[0], edata[1], device2.create_view(queuing_mode_automatic));

			// let the kernel initialize data;
			extent<rank> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<_type> opt1(e1.size());
			opt1 = src1;
			vector<_type> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return false;
			}
			printf ("Passed for rank %d\n", rank);
			rank_step = rank;
		}
		if (_rank > 2)
		{
			const int rank = 3;
			rank_step = rank;
			array<_type, rank> src1(edata[0], edata[1], edata[2], device1.create_view(queuing_mode_immediate));
			array<_type, rank> src2(edata[0], edata[1], edata[2], device2.create_view(queuing_mode_automatic));

			// let the kernel initialize data;
			extent<rank> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<_type> opt1(e1.size());
			opt1 = src1;
			vector<_type> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return false;
			}
			printf ("Passed for rank %d\n", rank);
			rank_step = rank;
		}

		{
			rank_step = _rank;
			extent<_rank> e1(edata);
			array<_type, _rank> src1(e1, device1.create_view(queuing_mode_immediate));
			array<_type, _rank> src2(e1, device2.create_view(queuing_mode_automatic));

			// let the kernel initialize data;
			parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<_type> opt1(e1.size());
			opt1 = src1;
			vector<_type> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return false;
			}
			printf ("Passed for rank %d\n", _rank);
			rank_step = _rank;
		}
		printf("Finished with device %d and %d\n", i, i);
	}

	return true;
}

runall_result test_main()
{ 
	iteration = 0;
	rank_step = 0;

	vector<accelerator> devices = get_available_devices(device_flags::INCLUDE_CPU);
	SKIP_IF(devices.size() < 2);

	try
	{
		test_feature<int, 5>(devices);
	}
    catch (std::exception e) 
    {
	if ((iteration > 0) || (rank_step > 0))
	{
		std::cout << "Partly executed test - iteration : " << iteration << " rank executed " << rank_step<< std::endl;
		return runall_fail;
	}
	else
	{
		return runall_pass;
	}
    }

    return runall_fail;
}

