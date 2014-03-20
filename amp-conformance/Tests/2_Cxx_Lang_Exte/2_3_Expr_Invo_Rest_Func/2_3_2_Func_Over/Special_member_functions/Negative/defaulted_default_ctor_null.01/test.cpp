// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test where default constructors cannot be defaulted due to incompatible restriction specifiers.</summary>
//#Expects: Error: test\.cpp\(113\) : .+ C2512:.*(\bA1\b)
//#Expects: Error: test\.cpp\(114\) : .+ C2512:.*(\bA2\b)
//#Expects: Error: test\.cpp\(115\) : .+ C2512:.*(\bA3\b)
//#Expects: Error: test\.cpp\(116\) : .+ C2512:.*(\bA4\b)
//#Expects: Error: test\.cpp\(117\) : .+ C2512:.*(\bA5\b)
//#Expects: Error: test\.cpp\(122\) : .+ C2512:.*(\bA1\b)
//#Expects: Error: test\.cpp\(123\) : .+ C2512:.*(\bA2\b)
//#Expects: Error: test\.cpp\(124\) : .+ C2512:.*(\bA3\b)
//#Expects: Error: test\.cpp\(125\) : .+ C2512:.*(\bA4\b)
//#Expects: Error: test\.cpp\(126\) : .+ C2512:.*(\bA5\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with base classes having user-defined default ctors
struct A1_base_1
{
	int i;
	A1_base_1() restrict(amp) {}
};
class A1_base_2
{
	int i;
public:
	A1_base_2() restrict(cpu) {}
};
class A1 : A1_base_1, public A1_base_2
{
	// no default ctor possible
};

// Class with data members having user-defined default ctors
union A2_member_1
{
	int i;
	A2_member_1() restrict(amp) {}
};
struct A2_member_2
{
	A2_member_2() restrict(cpu) {}
};
class A2
{
	A2_member_1 m1;
	A2_member_2 m2;
	// no default ctor possible
};

// Class with base class and data member having user-defined default ctors
struct A3_base_1
{
	int i;
	A3_base_1() restrict(cpu) {}
};
class A3_member_1
{
public:
	A3_member_1() restrict(amp) {}
};
struct A3 : A3_base_1
{
	A3_member_1 m1;
	// no default ctor possible
};

// Classes with user-defined dtor and base class and data member having user-defined default ctors
class A4_base_1
{
	int i;
public:
	A4_base_1() restrict(cpu) {}
};
union A4_member_1
{
	A4_member_1() restrict(cpu) {}
};
class A4 : public A4_base_1
{
	A4_member_1 m1;
public:
	~A4() restrict(amp) {}
	// no default ctor possible
};

class A5_base_1
{
	int i;
public:
	A5_base_1() restrict(amp) {}
};
union A5_member_1
{
	A5_member_1() restrict(amp) {}
};
class A5 : public A5_base_1
{
	A5_member_1 m1;
public:
	~A5() restrict(cpu) {}
	// no default ctor possible
};

void f() restrict(cpu)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
	A5 a5;
}

void f() restrict(amp)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
	A5 a5;
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

