// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test whether defaulted default constructors have restrict(cpu,amp) specifiers.</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

// Empty class with no base classes nor user-defined dtor
class A1
{
	// defaulted: A1() restrict(cpu,amp)
};

// Empty class with no base classes and user-defined dtor
union A2
{
public:
	~A2() restrict(cpu,amp) {}
	// defaulted: A2() restrict(cpu,amp)
};

// Empty class with base class having defaulted default ctor
class A3_base {};
class A3 : A3_base
{
	// defaulted: A3() restrict(cpu,amp)
};

// Empty class with base class having user-defined default ctor
struct A4_base
{
	A4_base() restrict(cpu,amp) {}
	~A4_base() restrict(cpu,amp) {}
};
struct A4 : public A4_base
{
	// defaulted: A4() restrict(cpu,amp)
};

// Empty class with base classes having both defaulted and user-defined default ctors
struct A5_base_1
{
	int i;
	A5_base_1() restrict(cpu) {}
	A5_base_1() restrict(amp) {}
};
class A5_base_2
{
	int i;
public:
	~A5_base_2() restrict(cpu,amp) {}
};
class A5 : A5_base_1, public A5_base_2
{
	// defaulted: A5() restrict(cpu,amp)
};

// Class with data members having both defaulted and user-defined default ctors
struct A6_member_1
{
	int i;
	A6_member_1() restrict(cpu,amp) {}
};
class A6_member_2
{
	int i;
public:
	~A6_member_2() restrict(cpu,amp) {}
};
class A6
{
	A6_member_1 m1;
	A6_member_2 m2;
	// defaulted: A6() restrict(cpu,amp)
};

// Class with base classes having both defaulted and user-defined default ctors,
// data members having both defaulted and user-defined default ctors
// and user-defined dtor.
class A7_base_1 { int i; };
class A7_base_2
{
	int i;
public:
	A7_base_2() restrict(cpu,amp) {}
};
class A7_member_1 { int i; };
struct A7_member_2
{
	int i;
	A7_member_2() restrict(cpu) {}
	A7_member_2() restrict(amp) {}
};
class A7 : A7_base_1, public A7_base_2
{
	A7_member_1 m1;
	A7_member_2 m2;
};

bool test() restrict(cpu,amp)
{
	A1 a1;
	A2 a2;
	A3 a3;
	A4 a4;
	A5 a5;
	A6 a6;
	A7 a7;
	return true; // Compile-time tests
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	result &= INVOKE_TEST_FUNC_ON_CPU_AND_GPU(av, test);
	return result;
}
