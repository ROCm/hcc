// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>User defined conversion function to a class type used in dual-restricted function.</summary>
// Note: This is a regression test for #388265.
#pragma warning(disable: 4189) // local variable is initialized but not referenced

struct X {};
struct Y {};
void func_X(const X&) restrict(cpu,amp) {}
void func_Y(const Y&) restrict(cpu,amp) {}

struct A
{
	operator X() restrict(cpu);
	operator Y() restrict(amp);
};

struct A2
{
	operator X() restrict(cpu);
	operator X() restrict(amp);
};

struct A3
{
	operator X() restrict(cpu,amp);
};

template <typename T>
struct B
{
	operator X() restrict(cpu);
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
template <> D::operator X() restrict(cpu) { return X(); }
// Note: cannot explicitely specialize amp-restricted conversion function, #391038

void f_A_cpu(A obj) restrict(cpu)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}

void f_A_amp(A obj) restrict(amp)
{
	Y i = obj;
	Y j = static_cast<Y>(obj);
	Y k = (Y)obj;
	func_Y(obj);
}

void f_A2(A2 obj) restrict(cpu,amp)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}

void f_A3(A3 obj) restrict(cpu,amp)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}

void f_B_X(B<X> obj) restrict(cpu,amp)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}

void f_C_X(C<X> obj) restrict(cpu,amp)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}

void f_D(D obj) restrict(cpu,amp)
{
	X i = obj;
	X j = static_cast<X>(obj);
	X k = (X)obj;
	func_X(obj);
}
