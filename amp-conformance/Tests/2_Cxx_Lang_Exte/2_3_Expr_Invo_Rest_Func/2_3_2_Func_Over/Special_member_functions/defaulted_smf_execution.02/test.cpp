// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether data member's SMFs are executed through defaulted ones in a containing class.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

array_view<int> make_empty_array_view(int& storage) restrict(cpu,amp)
{
	return array_view<int>(1, &storage);
}

struct cpu_and_amp
{
	cpu_and_amp() restrict(cpu,amp)
		: default_ctor_called(1)
		, copy_ctor_called(0)
		, copy_assign_op_called(0)
		, dtor_called_av(make_empty_array_view(mock_storage))
	{
	}

	cpu_and_amp(const cpu_and_amp&) restrict(cpu,amp)
		: default_ctor_called(0)
		, copy_ctor_called(1)
		, copy_assign_op_called(0)
		, dtor_called_av(make_empty_array_view(mock_storage))
	{
	}

	cpu_and_amp& operator=(const cpu_and_amp&) restrict(cpu,amp)
	{
		copy_assign_op_called++;
		return *this;
	}

	~cpu_and_amp() restrict(cpu,amp)
	{
		dtor_called_av[0]++;
	}

	int default_ctor_called,
		copy_ctor_called,
		copy_assign_op_called;
	int mock_storage;
	array_view<int> dtor_called_av;
};

struct member_cpu_and_amp
{
	cpu_and_amp m;
	// defaulted: member_cpu_and_amp() restrict(cpu,amp)
	// defaulted: member_cpu_and_amp(const member_cpu_and_amp&) restrict(cpu,amp)
	// defaulted: member_cpu_and_amp& operator=(const member_cpu_and_amp&) restrict(cpu,amp)
	// defaulted: ~member_cpu_and_amp() restrict(cpu,amp)
};

struct cpu_or_amp
{
	cpu_or_amp() restrict(cpu)
		: default_ctor_called_cpu(1)
		, default_ctor_called_amp(0)
		, copy_ctor_called_cpu(0)
		, copy_ctor_called_amp(0)
		, copy_assign_op_called_cpu(0)
		, copy_assign_op_called_amp(0)
	{
	}

	cpu_or_amp() restrict(amp)
		: default_ctor_called_cpu(0)
		, default_ctor_called_amp(1)
		, copy_ctor_called_cpu(0)
		, copy_ctor_called_amp(0)
		, copy_assign_op_called_cpu(0)
		, copy_assign_op_called_amp(0)
	{
	}

	cpu_or_amp(const cpu_or_amp&) restrict(cpu)
		: default_ctor_called_cpu(0)
		, default_ctor_called_amp(0)
		, copy_ctor_called_cpu(1)
		, copy_ctor_called_amp(0)
		, copy_assign_op_called_cpu(0)
		, copy_assign_op_called_amp(0)
	{
	}

	cpu_or_amp(const cpu_or_amp&) restrict(amp)
		: default_ctor_called_cpu(0)
		, default_ctor_called_amp(0)
		, copy_ctor_called_cpu(0)
		, copy_ctor_called_amp(1)
		, copy_assign_op_called_cpu(0)
		, copy_assign_op_called_amp(0)
	{
	}

	cpu_or_amp& operator=(const cpu_or_amp&) restrict(cpu)
	{
		copy_assign_op_called_cpu++;
		return *this;
	}

	cpu_or_amp& operator=(const cpu_or_amp&) restrict(amp)
	{
		copy_assign_op_called_amp++;
		return *this;
	}

	// Note: cannot declare separate destructors.

	int default_ctor_called_cpu,
		default_ctor_called_amp,
		copy_ctor_called_cpu,
		copy_ctor_called_amp,
		copy_assign_op_called_cpu,
		copy_assign_op_called_amp;
};

struct member_cpu_or_amp
{
	cpu_or_amp m;
	// defaulted: member_cpu_or_amp() restrict(cpu,amp)
	// defaulted: member_cpu_or_amp(const member_cpu_or_amp&) restrict(cpu,amp)
	// defaulted: member_cpu_or_amp& operator=(const member_cpu_or_amp&) restrict(cpu,amp)
	// defaulted: ~member_cpu_or_amp() restrict(cpu,amp)
};

struct amp
{
	amp() restrict(cpu,amp)
		: default_ctor_called(1)
		, copy_ctor_called(0)
		, copy_assign_op_called(0)
		, dtor_called_av(make_empty_array_view(mock_storage))
	{
	}

	amp(const amp&) restrict(cpu,amp)
		: default_ctor_called(0)
		, copy_ctor_called(1)
		, copy_assign_op_called(0)
		, dtor_called_av(make_empty_array_view(mock_storage))
	{
	}

	amp& operator=(const amp&) restrict(cpu,amp)
	{
		copy_assign_op_called++;
		return *this;
	}

	~amp() restrict(cpu,amp)
	{
		dtor_called_av[0]++;
	}

	int default_ctor_called,
		copy_ctor_called,
		copy_assign_op_called;
	int mock_storage;
	array_view<int> dtor_called_av;
};

struct member_amp
{
	amp m;
	// defaulted: member_amp() restrict(amp)
	// defaulted: member_amp(const member_amp&) restrict(amp)
	// defaulted: member_amp& operator=(const member_amp&) restrict(amp)
	// defaulted: ~member_amp() restrict(amp)
};

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
	runall_result result;

	// cpu,amp
	int mcaa_dtor_called = 0;
	{
		member_cpu_and_amp mcaa; // default ctor
		result &= REPORT_RESULT(mcaa.m.default_ctor_called == 1);
		result &= REPORT_RESULT(mcaa.m.copy_ctor_called == 0);
		result &= REPORT_RESULT(mcaa.m.copy_assign_op_called == 0);

		member_cpu_and_amp mcaa_copy(mcaa); // copy ctor
		result &= REPORT_RESULT(mcaa_copy.m.default_ctor_called == 0);
		result &= REPORT_RESULT(mcaa_copy.m.copy_ctor_called == 1);
		result &= REPORT_RESULT(mcaa_copy.m.copy_assign_op_called == 0);

		mcaa.m.default_ctor_called = 0;
		mcaa = mcaa_copy; // copy assignment operator
		result &= REPORT_RESULT(mcaa.m.default_ctor_called == 0);
		result &= REPORT_RESULT(mcaa.m.copy_ctor_called == 0);
		result &= REPORT_RESULT(mcaa.m.copy_assign_op_called == 1);

		mcaa.m.dtor_called_av = array_view<int>(1, &mcaa_dtor_called);
	} // dtor
	result &= REPORT_RESULT(mcaa_dtor_called == 1);

	// cpu|amp
	{
		member_cpu_or_amp mcoa; // default ctor
		result &= REPORT_RESULT(mcoa.m.default_ctor_called_cpu == 1);
		result &= REPORT_RESULT(mcoa.m.default_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa.m.copy_ctor_called_cpu == 0);
		result &= REPORT_RESULT(mcoa.m.copy_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa.m.copy_assign_op_called_cpu == 0);
		result &= REPORT_RESULT(mcoa.m.copy_assign_op_called_amp == 0);

		member_cpu_or_amp mcoa_copy(mcoa); // copy ctor
		result &= REPORT_RESULT(mcoa_copy.m.default_ctor_called_cpu == 0);
		result &= REPORT_RESULT(mcoa_copy.m.default_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa_copy.m.copy_ctor_called_cpu == 1);
		result &= REPORT_RESULT(mcoa_copy.m.copy_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa_copy.m.copy_assign_op_called_cpu == 0);
		result &= REPORT_RESULT(mcoa_copy.m.copy_assign_op_called_amp == 0);

		mcoa.m.default_ctor_called_cpu = 0;
		mcoa.m.default_ctor_called_amp = 0;
		mcoa = mcoa_copy; // copy assignment operator
		result &= REPORT_RESULT(mcoa.m.default_ctor_called_cpu == 0);
		result &= REPORT_RESULT(mcoa.m.default_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa.m.copy_ctor_called_cpu == 0);
		result &= REPORT_RESULT(mcoa.m.copy_ctor_called_amp == 0);
		result &= REPORT_RESULT(mcoa.m.copy_assign_op_called_cpu == 1);
		result &= REPORT_RESULT(mcoa.m.copy_assign_op_called_amp == 0);
	}

	int amp_result_[3] = {1, 1, 1};
	array_view<int> amp_result(3, amp_result_);
	parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(cpu,amp)
	{
		// cpu,amp
		int mcaa_dtor_called = 0;
		{
			member_cpu_and_amp mcaa; // default ctor
			amp_result[0] &= (mcaa.m.default_ctor_called == 1);
			amp_result[0] &= (mcaa.m.copy_ctor_called == 0);
			amp_result[0] &= (mcaa.m.copy_assign_op_called == 0);

			member_cpu_and_amp mcaa_copy(mcaa); // copy ctor
			amp_result[0] &= (mcaa_copy.m.default_ctor_called == 0);
			amp_result[0] &= (mcaa_copy.m.copy_ctor_called == 1);
			amp_result[0] &= (mcaa_copy.m.copy_assign_op_called == 0);

			mcaa.m.default_ctor_called = 0;
			mcaa = mcaa_copy; // copy assignment operator
			amp_result[0] &= (mcaa.m.default_ctor_called == 0);
			amp_result[0] &= (mcaa.m.copy_ctor_called == 0);
			amp_result[0] &= (mcaa.m.copy_assign_op_called == 1);

			mcaa.m.dtor_called_av = array_view<int>(1, &mcaa_dtor_called);
		} // dtor
		amp_result[0] &= (mcaa_dtor_called == 1);

		// cpu|amp
		{
			member_cpu_or_amp mcoa; // default ctor
			amp_result[1] &= (mcoa.m.default_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa.m.default_ctor_called_amp == 1);
			amp_result[1] &= (mcoa.m.copy_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa.m.copy_ctor_called_amp == 0);
			amp_result[1] &= (mcoa.m.copy_assign_op_called_cpu == 0);
			amp_result[1] &= (mcoa.m.copy_assign_op_called_amp == 0);

			member_cpu_or_amp mcoa_copy(mcoa); // copy ctor
			amp_result[1] &= (mcoa_copy.m.default_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa_copy.m.default_ctor_called_amp == 0);
			amp_result[1] &= (mcoa_copy.m.copy_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa_copy.m.copy_ctor_called_amp == 1);
			amp_result[1] &= (mcoa_copy.m.copy_assign_op_called_cpu == 0);
			amp_result[1] &= (mcoa_copy.m.copy_assign_op_called_amp == 0);

			mcoa.m.default_ctor_called_cpu = 0;
			mcoa.m.default_ctor_called_amp = 0;
			mcoa = mcoa_copy; // copy assignment operator
			amp_result[1] &= (mcoa.m.default_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa.m.default_ctor_called_amp == 0);
			amp_result[1] &= (mcoa.m.copy_ctor_called_cpu == 0);
			amp_result[1] &= (mcoa.m.copy_ctor_called_amp == 0);
			amp_result[1] &= (mcoa.m.copy_assign_op_called_cpu == 0);
			amp_result[1] &= (mcoa.m.copy_assign_op_called_amp == 1);
		}

		// amp
		int ma_dtor_called = 0;
		{
			member_amp ma; // default ctor
			amp_result[2] &= (ma.m.default_ctor_called == 1);
			amp_result[2] &= (ma.m.copy_ctor_called == 0);
			amp_result[2] &= (ma.m.copy_assign_op_called == 0);

			member_amp ma_copy(ma); // copy ctor
			amp_result[2] &= (ma_copy.m.default_ctor_called == 0);
			amp_result[2] &= (ma_copy.m.copy_ctor_called == 1);
			amp_result[2] &= (ma_copy.m.copy_assign_op_called == 0);

			ma.m.default_ctor_called = 0;
			ma = ma_copy; // copy assignment operator
			amp_result[2] &= (ma.m.default_ctor_called == 0);
			amp_result[2] &= (ma.m.copy_ctor_called == 0);
			amp_result[2] &= (ma.m.copy_assign_op_called == 1);

			ma.m.dtor_called_av = array_view<int>(1, &ma_dtor_called);
		} // dtor
		amp_result[2] &= (ma_dtor_called == 1);
	});

	result &= REPORT_RESULT(amp_result[0] == 1);
	result &= REPORT_RESULT(amp_result[1] == 1);
	result &= REPORT_RESULT(amp_result[2] == 1);
	return result;
}
