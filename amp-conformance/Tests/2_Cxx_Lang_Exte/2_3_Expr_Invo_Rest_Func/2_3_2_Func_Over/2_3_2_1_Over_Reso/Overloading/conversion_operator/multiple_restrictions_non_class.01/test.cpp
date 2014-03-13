// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>User defined conversion function to a non-class type used in dual-restricted function.</summary>
// Note: This is a regression test for #388265.
#pragma warning(disable: 4552 4189) // '-' : operator has no effect; local variable is initialized but not referenced

struct A
{
	operator int() restrict(cpu);
	operator long() restrict(amp);
};

struct A2
{
	operator int() restrict(cpu);
	operator int() restrict(amp);
};

struct A3
{
	operator int() restrict(cpu,amp);
};

template <typename T>
struct B
{
	operator int() restrict(cpu);
	operator T() restrict(amp);
};

template <typename T>
struct C
{
	operator T() restrict(cpu);

	template <typename U>
	operator U() restrict(amp);
};

struct D
{
	template <typename T>
	operator T() restrict(cpu);

	template <typename T>
	operator T() restrict(amp);
};
template <> D::operator int() restrict(cpu) { return 0; }
// Note: cannot explicitely specialize amp-restricted conversion function, #391038

void f_A_cpu(A obj) restrict(cpu)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_A_amp(A obj) restrict(amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_A2(A2 obj) restrict(cpu,amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_A3(A3 obj) restrict(cpu,amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_B_int(B<int> obj) restrict(cpu,amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_C_int(C<int> obj) restrict(cpu,amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	int f = obj;
	auto x = obj + 1;
	auto y = obj * obj;
	auto z = 1l * obj;
	obj - 0;
	-obj;
}

void f_D(D obj) restrict(cpu,amp)
{
	int i = obj;
	int j = static_cast<int>(obj);
	int k = (int)obj;
	long f = obj;
	auto x = (int)obj + 1; // Explicit cast due to FE bug #391478 (also in the following expressions)
	auto y = (int)obj * (int)obj;
	auto z = 1l * (long)obj;
	(int)obj - 0;
	-(int)obj;
}
