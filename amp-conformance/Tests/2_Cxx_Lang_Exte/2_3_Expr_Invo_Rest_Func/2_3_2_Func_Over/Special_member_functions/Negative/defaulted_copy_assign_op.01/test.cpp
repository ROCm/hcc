// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Negative counterpart to the test verifying copy assignment operators with restrict(cpu,amp), check constness of the parameter</summary>
//#Expects: Error: test\.cpp\(112\) : .+ C2679:.*(\bconst A3\b)
//#Expects: Error: test\.cpp\(116\) : .+ C3930:.*(\bA5::operator =)
//#Expects: Error: test\.cpp\(120\) : .+ C2679:.*(\bconst A7\b)
//#Expects: Error: test\.cpp\(124\) : .+ C2679:.*(\bconst A11\b)
//#Expects: Error: test\.cpp\(131\) : .+ C2679:.*(\bconst A3\b)
//#Expects: Error: test\.cpp\(135\) : .+ C3930:.*(\bA6::operator =)
//#Expects: Error: test\.cpp\(139\) : .+ C2679:.*(\bconst A7\b)
//#Expects: Error: test\.cpp\(143\) : .+ C3930:.*(\bA10::operator =)
//#Expects: Error: test\.cpp\(147\) : .+ C2679:.*(\bconst A11\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty classes with base class having user-defined copy op=
struct A3_base
{
	A3_base& operator=(A3_base&) restrict(cpu,amp) { return *this; }
};
struct A3 : public A3_base
{
	// defaulted: A3& operator=(A3&) restrict(cpu,amp)
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

void test_cpu() restrict(cpu)
{
	A3 a3l;
	const A3 a3r;
	a3l = a3r;

	A5 a5l;
	const A5 a5r;
	a5l = a5r;

	A7 a7l;
	const A7 a7r;
	a7l = a7r;

	A11 a11l;
	const A11 a11r;
	a11l = a11r;
}

void test_amp() restrict(amp)
{
	A3 a3l
	const A3 a3r;
	a3l = a3r;

	A6 a6l;
	const A6 a6r;
	a6l = a6r;

	A7 a7l;
	const A7 a7r;
	a7l = a7r;

	A10 a10l;
	const A10 a10r;
	a10l = a10r;

	A11 a11l;
	const A11 a11r;
	a11l = a11r;
}

runall_result test_main()
{
	return runall_fail; // Should not compile
}

