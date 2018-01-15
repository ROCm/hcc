// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test where destructors cannot be defaulted due to incompatible restriction specifiers.</summary>
//#Expects: Error: test\.cpp\(91\) : .+ C2512:.*(\bA1\b)
//#Expects: Error: test\.cpp\(92\) : .+ C2512:.*(\bA2\b)
//#Expects: Error: test\.cpp\(93\) : .+ C2512:.*(\bA3\b)
//#Expects: Error: test\.cpp\(94\) : .+ C2512:.*(\bA4\b)
//#Expects: Error: test\.cpp\(99\) : .+ C2512:.*(\bA1\b)
//#Expects: Error: test\.cpp\(100\) : .+ C2512:.*(\bA2\b)
//#Expects: Error: test\.cpp\(101\) : .+ C2512:.*(\bA3\b)
//#Expects: Error: test\.cpp\(102\) : .+ C2512:.*(\bA4\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with base classes having user-defined dtors
struct A1_base_1
{
	int i;
	~A1_base_1() restrict(amp) {}
};
class A1_base_2
{
	int i;
public:
	~A1_base_2() restrict(cpu) {}
};
class A1 : A1_base_1, public A1_base_2
{
public:
	// no dtor possible thus no default ctor possible
};

// Class with data members having user-defined dtors
union A2_member_1
{
	int i;
	~A2_member_1() restrict(amp) {}
};
struct A2_member_2
{
	~A2_member_2() restrict(cpu) {}
};
class A2
{
	A2_member_1 m1;
	A2_member_2 m2;
	// no dtor possible thus no default ctor possible
};

// Classes with base class having user-defined dtor and data member having user-defined dtor
struct A3_base_1
{
	int i;
	~A3_base_1() restrict(cpu) {}
};
class A3_member_1
{
public:
	~A3_member_1() restrict(amp) {}
};
struct A3 : A3_base_1
{
	A3_member_1 m1;
	// no dtor possible thus no default ctor possible
};

class A4_base_1
{
	int i;
public:
	~A4_base_1() restrict(amp) {}
};
union A4_member_1
{
	~A4_member_1() restrict(cpu) {}
};
class A4 : public A4_base_1
{
	A4_member_1 m1;
	// no dtor possible thus no default ctor possible
};

void f() restrict(cpu)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
}

void f() restrict(amp)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

