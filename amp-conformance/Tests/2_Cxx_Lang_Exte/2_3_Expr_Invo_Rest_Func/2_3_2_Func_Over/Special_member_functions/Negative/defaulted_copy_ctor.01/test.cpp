// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Negative counterpart to the test verifying copy constructors with restrict(cpu,amp), check constness of the parameter</summary>
//#Expects: Error: test\.cpp\(118\) : .+ C2558:.*(\bA4\b)
//#Expects: Error: test\.cpp\(121\) : .+ C3930:.*(\bA6::A6\b)
//#Expects: Error: test\.cpp\(124\) : .+ C2558:.*(\bA8\b)
//#Expects: Error: test\.cpp\(127\) : .+ C2558:.*(\bA13\b)
//#Expects: Error: test\.cpp\(133\) : .+ C2558:.*(\bA4\b)
//#Expects: Error: test\.cpp\(136\) : .+ C3930:.*(\bA7::A7\b)
//#Expects: Error: test\.cpp\(139\) : .+ C2558:.*(\bA8\b)
//#Expects: Error: test\.cpp\(142\) : .+ C3930:.*(\bA11::A11\b)
//#Expects: Error: test\.cpp\(145\) : .+ C2558:.*(\bA13\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

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

void test_cpu() restrict(cpu)
{
	const A4 a4;
	A4 a4c(a4);

	const A6 a6;
	A6 a6c(a6);

	const A8 a8;
	A8 a8c(a8);

	const A13 a13;
	A13 a13c(a13);
}

void test_amp() restrict(amp)
{
	const A4 a4;
	A4 a4c(a4);

	const A7 a7;
	A7 a7c(a7);

	const A8 a8;
	A8 a8c(a8);

	const A11 a11;
	A11 a11c(a11);

	const A13 a13;
	A13 a13c(a13);
}

runall_result test_main()
{
	return runall_fail; // Should not compile
}

