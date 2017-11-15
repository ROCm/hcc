// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test where copy assignment operators cannot be defaulted due to incompatible restriction specifiers.</summary>
//#Expects: Error: test\.cpp\(88\) : .+ C2582:.*(\bA1\b)
//#Expects: Error: test\.cpp\(92\) : .+ C2582:.*(\bA2\b)
//#Expects: Error: test\.cpp\(96\) : .+ C2582:.*(\bA3\b)
//#Expects: Error: test\.cpp\(111\) : .+ C2582:.*(\bA1\b)
//#Expects: Error: test\.cpp\(115\) : .+ C2582:.*(\bA2\b)
//#Expects: Error: test\.cpp\(119\) : .+ C2582:.*(\bA3\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with base classes having user-defined copy op=
struct A1_base_1
{
	int i;
	A1_base_1& operator=(const A1_base_1&) restrict(cpu) { return *this; }
};
class A1_base_2
{
	int i;
public:
	A1_base_2& operator=(const A1_base_2&) restrict(amp) { return *this; }
};
struct A1 : A1_base_1, A1_base_2
{
	// no copy op= possible
};

// Class with data members having user-defined copy op=
union A2_member_1
{
	int i;
	A2_member_1& operator=(const A2_member_1&) restrict(amp) { return *this; }
};
struct A2_member_2
{
	A2_member_2& operator=(const A2_member_2&) restrict(cpu) { return *this;}
};
class A2
{
	A2_member_1 m1;
	A2_member_2 m2;
	// no copy op= possible
};

// Class with base class and data member having user-defined copy op=
struct A3_base_1
{
	int i;
	A3_base_1& operator=(const A3_base_1&) restrict(cpu) { return *this; }
};
class A3_member_1
{
public:
	A3_member_1& operator=(const A3_member_1&) restrict(amp) { return *this; }
};
struct A3 : A3_base_1
{
	A3_member_1 m1;
	// no copy op= possible
};

// Classes with move assignment operators
struct A4
{
	A4& operator=(A4&&) restrict(cpu) { return *this; }
	// copy op= deleted
};

struct A5
{
	A5& operator=(A5&&) restrict(amp) { return *this; }
	// copy op= deleted
};

void f() restrict(cpu)
{
	A1 a1l;
	const A1 a1r;
	a1l = a1r;

	A2 a2l;
	const A2 a2r;
	a2l = a2r;

	A3 a3l;
	const A3 a3r;
	a3l = a3r;

	A4 a4l;
	const A4 a4r;
	a4l = a4r;

	A5 a5l;
	const A5 a5r;
	a5l = a5r;
}

void f() restrict(amp)
{
	A1 a1l;
	const A1 a1r;
	a1l = a1r;

	A2 a2l;
	const A2 a2r;
	a2l = a2r;

	A3 a3l;
	const A3 a3r;
	a3l = a3r;

	A4 a4l;
	const A4 a4r;
	a4l = a4r;

	A5 a5l;
	const A5 a5r;
	a5l = a5r;
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

