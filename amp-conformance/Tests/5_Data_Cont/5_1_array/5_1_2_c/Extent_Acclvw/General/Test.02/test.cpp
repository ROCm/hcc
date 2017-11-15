// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator specialized constructors - same device same default view.</summary>

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
int test_feature()
{
	vector<accelerator> devices = get_available_devices(device_flags::NOT_SPECIFIED);
	if(devices.empty()) {
		Log(LogType::Warning, true) << "No device exists." << std::endl;
		return runall_skip;
	}
	Log(LogType::Info, true) << "Found " << devices.size() << " devices." << std::endl;

	int edata[_rank];
	for (int i = 0; i < _rank; i++)
		edata[i] = 3;

	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator device1 = devices[i];
		accelerator device2 = devices[i];

		{
			extent<_rank> e1(edata);
			array<_type, _rank> src1(e1, device1.get_default_view());
			array<_type, _rank> src2(e1, device2.get_default_view());

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
					return runall_fail;
			}
		}

		if (_rank > 0)
		{
			const int rank = 1;
			array<_type, rank> src1(edata[0], device1.get_default_view());
			array<_type, rank> src2(edata[0], device2.get_default_view());

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
					return runall_fail;
			}
		}
		if (_rank > 1)
		{
			const int rank = 2;
			array<_type, rank> src1(edata[0], edata[1], device1.get_default_view());
			array<_type, rank> src2(edata[0], edata[1], device2.get_default_view());

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
					return runall_fail;
			}
		}
		if (_rank > 2)
		{
			const int rank = 3;
			array<_type, rank> src1(edata[0], edata[1], edata[2], device1.get_default_view());
			array<_type, rank> src2(edata[0], edata[1], edata[2], device2.get_default_view());

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
					return runall_fail;
			}
		}

		printf("Finished with device %zu \n", i);
	}

	return runall_pass;
}

template<int _rank>
int test_feature()
{
	// For doubles, we require limited double support
	vector<accelerator> devices = get_available_devices(device_flags::LIMITED_DOUBLE);
	if(devices.empty()) {
		Log(LogType::Warning, true) << "No device exists with limited double support." << std::endl;
		return runall_skip;
	}
	Log(LogType::Info, true) << "Found " << devices.size() << " devices." << std::endl;

	int edata[_rank];
	for (int i = 0; i < _rank; i++)
		edata[i] = 3;

	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator device1 = devices[i];
		accelerator device2 = devices[i];

		{
			extent<_rank> e1(edata);
			array<double, _rank> src1(e1, device1.get_default_view());
			array<double, _rank> src2(e1, device2.get_default_view());

			// let the kernel initialize data;
			parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<double> opt1(e1.size());
			opt1 = src1;
			vector<double> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return runall_fail;
			}
		}

		if (_rank > 0)
		{
			const int rank = 1;
			array<double, rank> src1(edata[0], device1.get_default_view());
			array<double, rank> src2(edata[0], device2.get_default_view());

			// let the kernel initialize data;
			extent<1> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<double> opt1(e1.size());
			opt1 = src1;
			vector<double> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return runall_fail;
			}
		}
		if (_rank > 1)
		{
			const int rank = 2;
			array<double, rank> src1(edata[0], edata[1], device1.get_default_view());
			array<double, rank> src2(edata[0], edata[1], device2.get_default_view());

			// let the kernel initialize data;
			extent<rank> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<double> opt1(e1.size());
			opt1 = src1;
			vector<double> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return runall_fail;
			}
		}
		if (_rank > 2)
		{
			const int rank = 3;
			array<double, rank> src1(edata[0], edata[1], edata[2], device1.get_default_view());
			array<double, rank> src2(edata[0], edata[1], edata[2], device2.get_default_view());

			// let the kernel initialize data;
			extent<rank> e1(edata);
			parallel_for_each(e1, [&](index<rank> idx) __GPU_ONLY
			{
				src1[idx] = _rank;
				src2[idx] = _rank;
			});

			// Copy data to CPU
			vector<double> opt1(e1.size());
			opt1 = src1;
			vector<double> opt2(e1.size());
			opt2 = src2;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if ((opt1[i] != _rank) || (opt2[i] != _rank))
					return runall_fail;
			}
		}

		printf("Finished with device %zu \n", i);
	}

	return runall_pass;
}

runall_result test_main()
{
	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 5>()));
	result &= REPORT_RESULT((test_feature<float, 5>()));
	result &= REPORT_RESULT((test_feature<5>())).treat_skip_as_pass();

    return result;
}

