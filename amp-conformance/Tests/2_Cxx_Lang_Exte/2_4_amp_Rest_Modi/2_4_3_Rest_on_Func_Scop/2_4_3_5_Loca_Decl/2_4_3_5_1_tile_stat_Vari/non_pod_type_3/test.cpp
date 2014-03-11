// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Type used for tile_static variable should have a trivial default constructor and a trivial destructor (positive).</summary>
#include <amptest.h>
#include <amptest_main.h>

// Trivial ctor and dtor, no data members
class A
{
};

// Trivial ctor and dtor, fundamental type data members
class B
{
	int i;
	float j;
};

// Trivial ctor and dtor, compound type data members
class C
{
	int i[3];
	A a;
	B b[4];
};

// Trivial ctor and dtor with inheritance
class D : public C
{
};

// Trivial ctor and dtor, data members and member functions
class E
{
public:
	void f() restrict(cpu,amp) { i = 1; }
	void g() restrict(amp) { f(); }
private:
	B b;
	int i;
	int j;
};

void func() restrict(amp)
{
	tile_static A ts_a;
	tile_static B ts_b;
	tile_static C ts_c;
	tile_static D ts_d;
	tile_static E ts_e;
}

runall_result test_main()
{
	return runall_pass; // Compile time tests
}
