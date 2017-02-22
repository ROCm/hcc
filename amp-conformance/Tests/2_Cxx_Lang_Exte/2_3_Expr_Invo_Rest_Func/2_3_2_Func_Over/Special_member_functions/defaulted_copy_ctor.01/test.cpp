// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted copy constructors have restrict(cpu,amp) specifiers.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with no base classes nor user-defined dtor
class A1
{
	// defaulted: A1(const A1&) restrict(cpu,amp)
};

// Empty class with no base classes and user-defined dtor
union A2
{
public:
	~A2() restrict(cpu,amp) {}
	// defaulted: A2(const A2&) restrict(cpu,amp)
};

// Empty class with base class having defaulted copy ctor
class A3_base {};
class A3 : A3_base
{
	// defaulted: A3(const A3&) restrict(cpu,amp)
};

// Empty classes with base class having user-defined copy ctors
struct A4_base
{
	A4_base() restrict(cpu,amp) {}
	A4_base(A4_base&) restrict(cpu,amp) {}
};
struct A4 : public A4_base
{
	// defaulted: A4(A4&) restrict(cpu,amp)
};

struct A5_base
{
	A5_base() restrict(cpu,amp) {}
	A5_base(const A5_base&) restrict(cpu,amp) {}
};
class A5 : public A5_base
{
	// defaulted: A5(const A5&) restrict(cpu,amp)
};

struct A6_base
{
	A6_base() restrict(cpu,amp) {}
	A6_base(A6_base&) restrict(cpu) {}
	A6_base(const A6_base&) restrict(amp) {}
};
struct A6 : A6_base
{
	// defaulted: A6(const A6&) restrict(cpu) - err on use
	// defaulted: A6(const A6&) restrict(amp)
};

class A7_base
{
public:
	A7_base() restrict(cpu,amp) {}
	A7_base(const A7_base&, int=0) restrict(cpu) {}
	A7_base(A7_base&, float=0.f, bool=true) restrict(amp) {}
};
class A7 : public A7_base
{
	// defaulted: A7(const A7&) restrict(cpu)
	// defaulted: A7(const A7&) restrict(amp) - err on use
};

struct A8_base
{
	A8_base() restrict(cpu,amp) {}
	A8_base(A8_base&) restrict(cpu) {}
	A8_base(A8_base&) restrict(amp) {}
};
struct A8 : A8_base
{
	// defaulted: A8(A8&) restrict(cpu,amp)
};

// Empty class with base classes having both defaulted and user-defined copy ctors
struct A9_base_1
{
	int i;
};
class A9_base_2
{
public:
	A9_base_2() restrict(cpu,amp) {}
	A9_base_2(const A9_base_2&) restrict(cpu) {}
	A9_base_2(const A9_base_2&) restrict(amp) {}
};
class A9 : A9_base_1, public A9_base_2
{
	// defaulted: A9(const A9&) restrict(cpu,amp)
};

// Classes with data members having both defaulted and user-defined copy ctors
struct A10_member_1
{
	int i;
	A10_member_1() restrict(cpu,amp) {}
	A10_member_1(const A10_member_1&) restrict(cpu,amp) {}
};
class A10_member_2
{
};
class A10
{
	A10_member_1 m1;
	A10_member_2 m2;
	// defaulted: A10(const A10&) restrict(cpu,amp)
};

class A11_member_1
{
	int i;
};
class A11_member_2
{
	int i;
public:
	A11_member_2() restrict(cpu,amp) {}
	A11_member_2(const A11_member_2&) restrict(cpu) {}
	A11_member_2(A11_member_2&) restrict(amp) {}
};
struct A11
{
	A11_member_1 m1;
	A11_member_2 m2;
	// defaulted: A11(const A11&) restrict(cpu)
	// defaulted: A11(const A11&) restrict(amp) - err on use
};

union A12_member_1
{
	A12_member_1() restrict(cpu,amp) {}
	A12_member_1(const A12_member_1&, int=0) restrict(cpu) {}
	A12_member_1(const A12_member_1&) restrict(amp) {}
};
class A12
{
	A12_member_1 m1;
	// defaulted: A12(const A12&) restrict(cpu,amp)
};

// Class with base classes having both defaulted and user-defined copy ctors,
// data members having both defaulted and user-defined copy ctors
// and user-defined dtor.
class A13_base_1 { int i; };
class A13_base_2
{
	int i;
public:
	A13_base_2() restrict(cpu,amp) {}
	A13_base_2(A13_base_2&) restrict(cpu,amp) {}
};
class A13_member_1 { int i; };
struct A13_member_2
{
	int i;
	A13_member_2() restrict(cpu,amp) {}
	A13_member_2(const A13_member_2&) restrict(cpu) {}
	A13_member_2(const A13_member_2&) restrict(amp) {}
};
class A13 : A13_base_1, public A13_base_2
{
	A13_member_1 m1;
	A13_member_2 m2;
	// defaulted: A13(A13&) restrict(cpu,amp)
};

bool test() restrict(cpu,amp)
{
	const A1 a1;
	A1 a1c(a1);

	const A2 a2;
	A2 a2c(a2);

	const A3 a3;
	A3 a3c(a3);

	A4 a4;
	A4 a4c(a4);

	const A5 a5;
	A5 a5c(a5);

	A8 a8;
	A8 a8c(a8);

	const A9 a9 = {};
	A9 a9c(a9);

	const A10 a10;
	A10 a10c(a10);

	const A12 a12;
	A12 a12c(a12);

	A13 a13;
	A13 a13c(a13);

	return true; // Compile-time tests
}

bool test_cpu() restrict(cpu)
{
	const A7 a7;
	A7 a7c(a7);

	const A11 a11 = {};
	A11 a11c(a11);

	return true; // Compile-time tests
}

int test_amp() restrict(amp)
{
	const A6 a6;
	A6 a6c(a6);

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
