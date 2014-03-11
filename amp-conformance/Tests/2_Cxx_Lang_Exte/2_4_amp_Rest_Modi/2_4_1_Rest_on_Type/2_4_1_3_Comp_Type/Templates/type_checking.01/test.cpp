// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Validate type checking for class template members.</summary>
// Note: this is a regression test for #396238.
#include "../type_checking_common.h"
#pragma warning(disable: 4101) // unreferenced local variable

template <int N, typename T>
struct obj_int
{
	int i;
};

// Instantiate non-amp-compatible objects in cpu function
void f_cpu()
{
	obj_N<0> o1;
	obj_N_T<5, char> o2;
	derived_10_char o3;
	derived_N_T<15, char> o4;
	member_20 o5;
	member_N<25> o6;

	obj_N<1>();
	obj_N_T<6, char>();
	derived_11_char();
	derived_N_T<16, char>();
	member_21();
	member_N<26>();
}

// Non-amp-compatible type used merely as template argument
void f_amp() restrict(amp)
{
	obj_int<0, char> o1;
	obj_int<1, char>();
}

void f_cpu_amp() restrict(cpu,amp)
{
	obj_int<2, char> o1;
	obj_int<3, char>();
}

// Default function arguments:
// - non-amp-compatible type for cpu function
// - non-amp-compatible type as template argument for amp function
void f_cpu_1(obj_N_T<7, char> = obj_N_T<7, char>())
{}

void f_cpu_call()
{
	f_cpu_1();
}

void f_amp_1(obj_int<4, char> = obj_int<4, char>()) restrict(amp)
{}

void f_amp_call() restrict(amp)
{
	f_amp_1();
}

void f_cpu_amp_1(obj_int<5, char> = obj_int<5, char>()) restrict(cpu,amp)
{}

void f_cpu_amp_call() restrict(cpu,amp)
{
	f_cpu_amp_1();
}

// Instantiate only to access static member function
template <int, typename T>
struct obj_static_N_T
{
	T m;
	static void f_amp() restrict(amp) {}
	static void f_cpu_amp() restrict(cpu,amp) {}
};

void f_obj_static_amp() restrict(amp)
{
	obj_static_N_T<0, char>::f_amp();
}

void f_obj_static_cpu_amp() restrict(cpu,amp)
{
	obj_static_N_T<1, char>::f_cpu_amp();
}
