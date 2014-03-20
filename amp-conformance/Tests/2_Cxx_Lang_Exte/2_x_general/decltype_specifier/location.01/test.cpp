// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify decltype in various syntactically correct locations.</summary>
#pragma warning(disable: 4101 4189) // unreferenced local variable; unreferenced initialized local variable
#pragma warning(disable: 4127 4930) // conditional expression is constant; prototype function not called

struct cpu_t
{
	operator bool() restrict(cpu,amp); // Req'd to define in 'if' condition
};
struct amp_t
{
	operator bool() restrict(cpu,amp); // Req'd to define in 'if' condition
	int i; // Req'd to satisfy alignment
};

cpu_t f() restrict(cpu);
amp_t f() restrict(amp);

// Multiple function declarators
void wrap_test_multdecl_1() restrict(amp), test_multdecl_1(decltype(f())); // expect: void test_multdecl_1(cpu_t) restrict(cpu)
void test_multdecl_1_verify()
{
	test_multdecl_1(cpu_t()); // verify
}

// Return type
decltype(f()) test_rt_1() restrict(amp); // expect: cpu_t test_rt_1() restrict(amp)
void test_rt_1_verify() restrict(amp)
{
	cpu_t r = test_rt_1(); // verify
}

void wrap_test_rt_2() restrict(amp)
{
	extern decltype(f()) test_rt_2() restrict(cpu); // expect: amp_t test_rt_2() restrict(cpu)
	[]() restrict(cpu)
	{
		amp_t r = test_rt_2(); // verify
	};
}

// Trailing return type
auto test_trt_1() restrict(cpu) -> decltype(f()); // expect: cpu_t test_3() restrict(cpu)
void test_trt_1_verify()
{
	cpu_t r = test_trt_1(); // verify
}

auto test_trt_2() restrict(amp) -> decltype(f()); // expect: amp_t test_trt_2() restrict(amp)
void test_trt_2_verify() restrict(amp)
{
	amp_t r = test_trt_2(); // verify
}

// Function parameter
void test_param_1(decltype(f())) restrict(amp); // expect: void test_param_1(cpu_t) restrict(amp)
void test_param_1_verify() restrict(amp)
{
	test_param_1(cpu_t()); // verify
}

void wrap_test_param_2() restrict(amp)
{
	extern void test_param_2(decltype(f())) restrict(cpu); // expect: void test_param_2(amp_t) restrict(cpu)
	[]() restrict(cpu)
	{
		test_param_2(amp_t()); // verify
	};
}

// Block declaration, if condition declaration, for and range-for declaration
void wrap_test_decl_1() restrict(cpu)
{
	decltype(f()) local; // expect: cpu_t local
	local = cpu_t(); // verify

	cpu_t local_2 = decltype(f())(); // expect: cpu_t(); verify

	if(decltype(f()) cond = cpu_t()) // expect: cpu_t cond; verify
	{}

	for(decltype(f()) it; false;) // expect: cpu_t it
	{
		it = cpu_t(); // verify
	}

	cpu_t arr[1];
	for(decltype(f()) it : arr) // expect cpu_t it; verify
	{}
}

void wrap_test_decl_2() restrict(amp)
{
	decltype(f()) local; // expect: amp_t local
	local = amp_t(); // verify

	amp_t local_2 = decltype(f())(); // expect: amp_t(); verify

	if(decltype(f()) cond = amp_t()) // expect: amp_t cond; verify
	{}

	for(decltype(f()) it; false;) // expect: amp_t it
	{
		it = amp_t(); // verify
	}

	amp_t arr[1];
	for(decltype(f()) it : arr) // expect amp_t it; verify
	{}
}

// Class member declaration
struct test_mem_1
{
	decltype(f()) member; // expect: cpu_t member
};
void test_mem_1_verify()
{
	cpu_t r = test_mem_1().member; // verify
}

void wrap_test_mem_2() restrict(amp)
{
	struct test_mem_2
	{
		decltype(f()) member; // expect: amp_t member
	};
	
	amp_t r = test_mem_2().member; // verify
}

// Enumeration base
short    f_integral() restrict(cpu);
unsigned f_integral() restrict(amp);

enum test_enum_1 : decltype(f_integral()); // expect: enum test_enum_1 : short
enum test_enum_1 : short; // verify

void wrap_test_enum_2() restrict(amp)
{
	enum test_enum_2 : decltype(f_integral()); // expect : enum test_enum_2 : unsigned
	enum test_enum_2 : unsigned; // verify
}
