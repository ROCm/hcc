// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted copy assignment operators have restrict(cpu,amp) specifiers.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with no base classes
class A1
{
	// defaulted: A1& operator=(const A1&) restrict(cpu,amp)
};

// Empty class with base class having defaulted copy op=
class A2_base {};
class A2 : A2_base
{
	// defaulted: A2& operator=(const A2&) restrict(cpu,amp)
};

// Empty classes with base class having user-defined copy op=
struct A3_base
{
	A3_base& operator=(A3_base&) restrict(cpu,amp) { return *this; }
};
struct A3 : public A3_base
{
	// defaulted: A3& operator=(A3&) restrict(cpu,amp)
};

struct A4_base
{
	A4_base& operator=(const A4_base&) restrict(cpu,amp) { return *this; }
};
class A4 : public A4_base
{
	// defaulted: A4& operator=(const A4&) restrict(cpu,amp)
};

struct A5_base
{
	A5_base& operator=(A5_base&) restrict(cpu) { return *this; }
	A5_base& operator=(const A5_base&) restrict(amp) { return *this; }
};
struct A5 : A5_base
{
	// defaulted: A5& operator=(const A5&) restrict(cpu) - err on use
	// defaulted: A5& operator=(const A5&) restrict(amp)
};

class A6_base
{
public:
	A6_base& operator=(const A6_base&) restrict(cpu) { return *this; }
	A6_base& operator=(A6_base&) restrict(amp) { return *this; }
};
class A6 : public A6_base
{
	// defaulted: A6& operator=(const A6&) restrict(cpu)
	// defaulted: A6& operator=(const A6&) restrict(amp) - err on use
};

struct A7_base
{
	A7_base& operator=(A7_base&) restrict(cpu) { return *this; }
	A7_base& operator=(A7_base&) restrict(amp) { return *this; }
};
struct A7 : A7_base
{
	// defaulted: A7& operator=(A7&) restrict(cpu,amp)
};

// Empty class with base classes having both defaulted and user-defined copy op=
struct A8_base_1
{
	int i;
};
class A8_base_2
{
public:
	A8_base_2& operator=(const A8_base_2&) restrict(cpu) { return *this; }
	A8_base_2& operator=(const A8_base_2&) restrict(amp) { return *this; }
};
class A8 : A8_base_1, public A8_base_2
{
	// defaulted: A8& operator=(const A8&) restrict(cpu,amp)
};

// Classes with data members having both defaulted and user-defined copy op=
struct A9_member_1
{
	int i;
	A9_member_1& operator=(const A9_member_1&) restrict(cpu,amp) { return *this; }
};
class A9_member_2
{
};
class A9
{
	A9_member_1 m1;
	A9_member_2 m2;
	// defaulted: A9& operator=(const A9&) restrict(cpu,amp)
};

class A10_member_1
{
	int i;
};
class A10_member_2
{
	int i;
public:
	A10_member_2& operator=(const A10_member_2&) restrict(cpu) { return *this; }
	A10_member_2& operator=(A10_member_2&) restrict(amp) { return *this; }
};
struct A10
{
	A10_member_1 m1;
	A10_member_2 m2;
	// defaulted: A10& operator=(const A10&) restrict(cpu)
	// defaulted: A10& operator=(const A10&) restrict(amp) - err on use
};

// Class with base classes having both defaulted and user-defined copy op=
// and data members having both defaulted and user-defined copy op=.
class A11_base_1 { int i; };
class A11_base_2
{
	int i;
public:
	A11_base_2& operator=(A11_base_2&) restrict(cpu,amp) { return *this; }
};
class A11_member_1 { int i; };
struct A11_member_2
{
	int i;
	A11_member_2& operator=(const A11_member_2&) restrict(cpu) { return *this; }
	A11_member_2& operator=(const A11_member_2&) restrict(amp) { return *this; }
};
class A11 : A11_base_1, public A11_base_2
{
	A11_member_1 m1;
	A11_member_2 m2;
	// defaulted: A11& operator=(A11&) restrict(cpu,amp)
};

bool test() restrict(cpu,amp)
{
	A1 a1l;
	const A1 a1r;
	a1l = a1r;

	A2 a2l;
	const A2 a2r;
	a2l = a2r;

	A3 a3l, a3r;
	a3l = a3r;

	A4 a4l;
	const A4 a4r;
	a4l = a4r;

	A7 a7l, a7r;
	a7l = a7r;

	A8 a8l;
	const A8 a8r = {};
	a8l = a8r;

	A9 a9l;
	const A9 a9r = {};
	a9l = a9r;

	A11 a11l, a11r;
	a11l = a11r;

	return true; // Compile-time tests
}

bool test_cpu() restrict(cpu)
{
	A6 a6l;
	const A6 a6r;
	a6l = a6r;

	A10 a10l;
	const A10 a10r = {};
	a10l = a10r;

	return true; // Compile-time tests
}

int test_amp() restrict(amp)
{
	A5 a5l;
	const A5 a5r;
	a5l = a5r;

	return 1; // Compile-time tests
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= REPORT_RESULT(INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, test));
	result &= REPORT_RESULT(test_cpu());
	result &= REPORT_RESULT(GPU_INVOKE(av, int, test_amp) == 1);
	return result;
}
