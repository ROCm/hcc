//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Read/write to a const array_view, test const-ness of indexing member functions.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

#pragma warning(push)
#pragma warning(disable : 4512)	// warning C4512: assignment operator could not be generated
class data_const
{
public:
	data_const(array_view<int, 1> output_1, array_view<int, 2> output_2, array_view<int, 3> output_3) : output_1(output_1), output_2(output_2), output_3(output_3) {};
	data_const(const data_const& rhs) restrict(cpu,amp) : output_1(rhs.output_1), output_2(rhs.output_2), output_3(rhs.output_3) {};

	void sync() const
	{
		output_1.synchronize();
		output_2.synchronize();
		output_3.synchronize();
	}

	const array_view<int, 1> output_1;
	const array_view<int, 2> output_2;
	const array_view<int, 3> output_3;
};
#pragma warning(pop)

class data_nonconst
{
public:
	data_nonconst(array_view<int, 1> output_1, array_view<int, 2> output_2, array_view<int, 3> output_3) : output_1(output_1), output_2(output_2), output_3(output_3) {};
	data_nonconst(const data_nonconst& rhs) restrict(cpu,amp) : output_1(rhs.output_1), output_2(rhs.output_2), output_3(rhs.output_3) {};
	~data_nonconst() restrict(cpu,amp) {}

	void sync() const
	{
		output_1.synchronize();
		output_2.synchronize();
		output_3.synchronize();
	}

	array_view<int, 1> output_1;
	array_view<int, 2> output_2;
	array_view<int, 3> output_3;
};

void function_value(const data_nonconst data) restrict(amp)
{
	data.output_1[index<1>(0)]			+= 0x00000001;
	data.output_1[int(0)]				+= 0x00000002;
	data.output_1(index<1>(0))			+= 0x00000004;
	data.output_1(int(0))				+= 0x00000008;
	data.output_2[index<2>(0,0)]		+= 0x00000001;
	data.output_2(index<2>(0,0))		+= 0x00000002;
	data.output_2(int(0),int(0))		+= 0x00000004;
	data.output_3[index<3>(0,0,0)]		+= 0x00000001;
	data.output_3(index<3>(0,0,0))		+= 0x00000002;
	data.output_3(int(0),int(0),int(0))	+= 0x00000004;
}

void function_ref(const data_nonconst& data) restrict(amp)
{
	data.output_1[index<1>(0)]			+= 0x00000001;
	data.output_1[int(0)]				+= 0x00000002;
	data.output_1(index<1>(0))			+= 0x00000004;
	data.output_1(int(0))				+= 0x00000008;
	data.output_2[index<2>(0,0)]		+= 0x00000001;
	data.output_2(index<2>(0,0))		+= 0x00000002;
	data.output_2(int(0),int(0))		+= 0x00000004;
	data.output_3[index<3>(0,0,0)]		+= 0x00000001;
	data.output_3(index<3>(0,0,0))		+= 0x00000002;
	data.output_3(int(0),int(0),int(0))	+= 0x00000004;
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();
	runall_result result;

	// Const member field in a function object
	{
		int output_1_;
		int output_2_;
		int output_3_;
		array_view<int, 1> output_1(1, &output_1_);
		array_view<int, 2> output_2(1, 1, &output_2_);
		array_view<int, 3> output_3(1, 1, 1, &output_3_);

		#pragma warning(push)
		#pragma warning(disable : 4512)	// warning C4512: assignment operator could not be generated
		class functor_
		{
		public:
			functor_(array_view<int, 1> output_1, array_view<int, 2> output_2, array_view<int, 3> output_3) : output_1(output_1), output_2(output_2), output_3(output_3) {};
			functor_(const functor_& rhs) : output_1(rhs.output_1), output_2(rhs.output_2), output_3(rhs.output_3) {};

			void operator()(index<1>) const restrict(cpu,amp)
			{
				output_1[index<1>(0)]			+= 0x00000001;
				output_1[int(0)]				+= 0x00000002;
				output_1(index<1>(0))			+= 0x00000004;
				output_1(int(0))				+= 0x00000008;
				output_2[index<2>(0,0)]			+= 0x00000001;
				output_2(index<2>(0,0))			+= 0x00000002;
				output_2(int(0),int(0))			+= 0x00000004;
				output_3[index<3>(0,0,0)]		+= 0x00000001;
				output_3(index<3>(0,0,0))		+= 0x00000002;
				output_3(int(0),int(0),int(0))	+= 0x00000004;
			}

			void sync()
			{
				output_1.synchronize();
				output_2.synchronize();
				output_3.synchronize();
			}

		private:
			const array_view<int, 1> output_1;
			const array_view<int, 2> output_2;
			const array_view<int, 3> output_3;
		} functor(output_1, output_2, output_3);
		#pragma warning(pop)

		// CPU
		// Note: one test for const-correctness in CPU context is enough, but for AMP we have
		// to check at least couple of code generation possibilites.
		output_1_ = 0;
		output_2_ = 0;
		output_3_ = 0;
		functor(index<1>(0));
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);

		// AMP
		output_1_ = 0;
		output_2_ = 0;
		output_3_ = 0;
		functor.sync();
		parallel_for_each(av, extent<1>(1), functor);
		functor.sync();
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);
	}

	// Const member field in a class
	{
		int output_1_ = 0;
		int output_2_ = 0;
		int output_3_ = 0;
		array_view<int, 1> output_1(1, &output_1_);
		array_view<int, 2> output_2(1, 1, &output_2_);
		array_view<int, 3> output_3(1, 1, 1, &output_3_);

		data_const data(output_1, output_2, output_3);
		parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(amp)
		{
			data.output_1[index<1>(0)]			+= 0x00000001;
			data.output_1[int(0)]				+= 0x00000002;
			data.output_1(index<1>(0))			+= 0x00000004;
			data.output_1(int(0))				+= 0x00000008;
			data.output_2[index<2>(0,0)]		+= 0x00000001;
			data.output_2(index<2>(0,0))		+= 0x00000002;
			data.output_2(int(0),int(0))		+= 0x00000004;
			data.output_3[index<3>(0,0,0)]		+= 0x00000001;
			data.output_3(index<3>(0,0,0))		+= 0x00000002;
			data.output_3(int(0),int(0),int(0))	+= 0x00000004;
		});
		data.sync();
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);
	}

	// Member field in a class in a const class (inherited constness)
	{
		int output_1_ = 0;
		int output_2_ = 0;
		int output_3_ = 0;
		array_view<int, 1> output_1(1, &output_1_);
		array_view<int, 2> output_2(1, 1, &output_2_);
		array_view<int, 3> output_3(1, 1, 1, &output_3_);

		const struct wrap_
		{
			wrap_(array_view<int, 1> output_1, array_view<int, 2> output_2, array_view<int, 3> output_3) : data(output_1, output_2, output_3) {}
			data_nonconst data;
		} wrap(output_1, output_2, output_3);
		parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(amp)
		{
			wrap.data.output_1[index<1>(0)]				+= 0x00000001;
			wrap.data.output_1[int(0)]					+= 0x00000002;
			wrap.data.output_1(index<1>(0))				+= 0x00000004;
			wrap.data.output_1(int(0))					+= 0x00000008;
			wrap.data.output_2[index<2>(0,0)]			+= 0x00000001;
			wrap.data.output_2(index<2>(0,0))			+= 0x00000002;
			wrap.data.output_2(int(0),int(0))			+= 0x00000004;
			wrap.data.output_3[index<3>(0,0,0)]			+= 0x00000001;
			wrap.data.output_3(index<3>(0,0,0))			+= 0x00000002;
			wrap.data.output_3(int(0),int(0),int(0))	+= 0x00000004;
		});
		wrap.data.sync();
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);
	}

	// Constness added on function parameter (pass by value)
	{
		int output_1_ = 0;
		int output_2_ = 0;
		int output_3_ = 0;
		array_view<int, 1> output_1(1, &output_1_);
		array_view<int, 2> output_2(1, 1, &output_2_);
		array_view<int, 3> output_3(1, 1, 1, &output_3_);

		data_nonconst data(output_1, output_2, output_3);
		parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(amp)
		{
			function_value(data);
		});
		data.sync();
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);
	}

	// Constness added on function parameter (pass by reference)
	{
		int output_1_ = 0;
		int output_2_ = 0;
		int output_3_ = 0;
		array_view<int, 1> output_1(1, &output_1_);
		array_view<int, 2> output_2(1, 1, &output_2_);
		array_view<int, 3> output_3(1, 1, 1, &output_3_);

		data_nonconst data(output_1, output_2, output_3);
		parallel_for_each(av, extent<1>(1), [=](index<1>) restrict(amp)
		{
			function_ref(data);
		});
		data.sync();
		result &= REPORT_RESULT(output_1_ == 0x0000000F);
		result &= REPORT_RESULT(output_2_ == 0x00000007);
		result &= REPORT_RESULT(output_3_ == 0x00000007);
	}

	return result;
}
