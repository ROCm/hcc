// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether SMF are defaulted with restrict(cpu,amp) for local class.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	class A
	{
		// defaulted: A() restrict(cpu,amp)
		// defaulted: A(const A&) restrict(cpu,amp)
		// defaulted: ~A() restrict(cpu,amp)
		// defaulted: A& operator=(const A&) restrict(cpu,amp)
	};

	const A a;
	A ac(a);
	ac = a;

	parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp)
	{
		const A a;
		A ac(a);
		ac = a;

		class B
		{
			// defaulted: B() restrict(cpu,amp)
			// defaulted: B(const B&) restrict(cpu,amp)
			// defaulted: ~B() restrict(cpu,amp)
			// defaulted: B& operator=(const B&) restrict(cpu,amp)
		};

		const B b;
		B bc(b);
		bc = b;

		[]() restrict(cpu)
		{
			// We cannot execute cpu from amp, but at least let's check definitions.
			const B b;
			B bc(b);
			bc = b;
		};
	});

	return runall_pass; // Compile-time tests
}
