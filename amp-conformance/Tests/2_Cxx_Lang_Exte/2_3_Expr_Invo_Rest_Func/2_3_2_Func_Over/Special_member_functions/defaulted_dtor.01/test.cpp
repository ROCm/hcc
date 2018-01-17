// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted destructors have restrict(cpu,amp) specifiers.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with no base classes
class A1
{
public:
	A1(int) restrict(cpu,amp) {}
	// defaulted: ~A1() restrict(cpu,amp)
};

// Empty class with base class having defaulted dtor
class A2_base {};
class A2 : A2_base
{
	// defaulted: ~A2() restrict(cpu,amp)
};

// Empty class with base class having user-defined dtor
struct A3_base
{
	~A3_base() restrict(cpu,amp) {}
};
struct A3 : public A3_base
{
	A3(float = 3.f) restrict(cpu,amp) {}
	// defaulted: ~A3() restrict(cpu,amp)
};

// Empty class with base classes having both defaulted and user-defined dtors
class A4_base_1 { int i; };
class A4_base_2
{
	int i;
public:
	~A4_base_2() restrict(cpu,amp) {}
};
class A4 : A4_base_1, public A4_base_2
{
	// defaulted: ~A4() restrict(cpu,amp)
};

// Class with scalar data members
union A5
{
	int i;
	float f;
	// defaulted: ~A5() restrict(cpu,amp)
};

// Class with data members having defaulted dtors
struct A6_member_1 {};
union A6_member_2
{
	int i;
	float f;
};
class A6
{
	A6_member_1 m1;
	A6_member_2 m2;
	// defaulted: ~A6() restrict(cpu,amp)
};

// Class with data members having user-defined dtors
struct A7_member_1
{
	~A7_member_1() restrict(cpu,amp) {}
};
class A7_member_2
{
public:
	~A7_member_2() restrict(cpu,amp) {}
	int i;
	float f;
};
class A7
{
	A7_member_1 m1;
	A7_member_2 m2;
	// defaulted: ~A7() restrict(cpu,amp)
};

// Class with base classes having both defaulted and user-defined dtors
// and data members having both defaulted and user-defined dtors
class A8_base_1 { int i; };
struct A8_base_2
{
	int i;
	~A8_base_2() restrict(cpu,amp) {}
};
class A8_member_1 { int i; };
struct A8_member_2
{
	int i;
	~A8_member_2() restrict(cpu,amp) {}
};
class A8 : protected A8_base_1, public A8_base_2
{
	A8_member_1 m1;
	A8_member_2 m2;
	// defaulted: ~A8() restrict(cpu,amp)
};

bool test() restrict(cpu,amp)
{
	A1 a1(1);
	A2 a2;
	A3 a3;
	A4 a4;
	A5 a5;
	A6 a6;
	A7 a7;
	A8 a8;
	return true; // Compile-time tests
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, test);
	return result;
}
