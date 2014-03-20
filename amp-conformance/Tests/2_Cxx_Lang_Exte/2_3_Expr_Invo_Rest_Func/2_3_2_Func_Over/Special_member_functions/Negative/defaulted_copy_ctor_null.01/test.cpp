// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test where copy constructors cannot be defaulted due to incompatible restriction specifiers.</summary>
//#Expects: Error: test\.cpp\(138\) : .+ C2558:.*(\bA1\b)
//#Expects: Error: test\.cpp\(141\) : .+ C2558:.*(\bA2\b)
//#Expects: Error: test\.cpp\(144\) : .+ C2558:.*(\bA3\b)
//#Expects: Error: test\.cpp\(147\) : .+ C2558:.*(\bA5\b)
//#Expects: Error: test\.cpp\(159\) : .+ C2558:.*(\bA1\b)
//#Expects: Error: test\.cpp\(162\) : .+ C2558:.*(\bA2\b)
//#Expects: Error: test\.cpp\(165\) : .+ C2558:.*(\bA3\b)
//#Expects: Error: test\.cpp\(168\) : .+ C2558:.*(\bA4\b)
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with base classes having user-defined copy ctors
struct A1_base_1
{
	int i;
	A1_base_1() restrict(cpu,amp) {}
	A1_base_1(const A1_base_1&) restrict(cpu) {}
};
class A1_base_2
{
	int i;
public:
	A1_base_2() restrict(cpu,amp) {}
	A1_base_2(const A1_base_2&) restrict(amp) {}
};
struct A1 : A1_base_1, A1_base_2
{
	// no copy ctor possible
};

// Class with data members having user-defined copy ctors
union A2_member_1
{
	int i;
	A2_member_1() restrict(cpu,amp) {}
	A2_member_1(const A2_member_1&) restrict(amp) {}
};
struct A2_member_2
{
	A2_member_2() restrict(cpu,amp) {}
	A2_member_2(const A2_member_2&) restrict(cpu) {}
};
class A2
{
	A2_member_1 m1;
	A2_member_2 m2;
	// no copy ctor possible
};

// Class with base class and data member having user-defined copy ctors
struct A3_base_1
{
	int i;
	A3_base_1() restrict(cpu,amp) {}
	A3_base_1(const A3_base_1&) restrict(cpu) {}
};
class A3_member_1
{
public:
	A3_member_1() restrict(cpu,amp) {}
	A3_member_1(const A3_member_1&) restrict(amp) {}
};
struct A3 : A3_base_1
{
	A3_member_1 m1;
	// no copy ctor possible
};

// Classes with user-defined dtor and base class and data member having user-defined copy ctors
class A4_base_1
{
	int i;
public:
	A4_base_1() restrict(cpu,amp) {}
	A4_base_1(const A4_base_1&) restrict(cpu) {}
};
union A4_member_1
{
	A4_member_1() restrict(cpu,amp) {}
	A4_member_1(const A4_member_1&) restrict(cpu) {}
};
class A4 : public A4_base_1
{
	A4_member_1 m1;
public:
	A4() restrict(amp) {}
	~A4() restrict(amp) {}
	// no copy ctor possible
};

class A5_base_1
{
	int i;
public:
	A5_base_1() restrict(cpu,amp) {}
	A5_base_1(const A5_base_1&) restrict(amp) {}
};
union A5_member_1
{
	A5_member_1() restrict(cpu,amp) {}
	A5_member_1(const A5_member_1&) restrict(amp) {}
};
class A5 : public A5_base_1
{
	A5_member_1 m1;
public:
	A5() restrict(cpu) {}
	~A5() restrict(cpu) {}
	// no copy ctor possible
};

/* post-Dev11 #345711
// Classes with move assignment operators
struct A6
{
	void operator=(A6&&) restrict(cpu) {}
	// copy constructor deleted
};

struct A7
{
	void operator=(A7&&) restrict(amp) {}
	// copy constructor deleted
};*/

void f() restrict(cpu)
{
	A1 a1;
	A1 a1c(a1);

	A2 a2;
	A2 a2c(a2);

	A3 a3;
	A3 a3c(a3);

	A5 a5;
	A5 a5c(a5);

	/*A6 a6;
	A6 a6c(a6);

	A7 a7;
	A7 a7c(a7);*/
}

void f() restrict(amp)
{
	A1 a1;
	A1 a1c(a1);

	A2 a2;
	A2 a2c(a2);

	A3 a3;
	A3 a3c(a3);

	A4 a4;
	A4 a4c(a4);

	/*A6 a6;
	A6 a6c(a6);

	A7 a7;
	A7 a7c(a7);*/
}

runall_result test_main()
{
	return runall_fail; // Should not compile.
}

