// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

// Empty class with base class having user-defined dtor
struct A1_base
{
	~A1_base() restrict(cpu) {}
};
struct A1 : public A1_base
{
	// defaulted: ~A1() restrict(cpu)
};

// Empty class with base classes having both defaulted and user-defined dtors
class A2_base_1 {};
class A2_base_2
{
public:
	~A2_base_2() restrict(cpu) {}
};
class A2 : A2_base_1, public A2_base_2
{
	// defaulted: ~A2() restrict(cpu)
};

// Empty class with base classes having user-defined dtors, one more restrictive than the other
class A3_base_1
{
public:
	~A3_base_1() restrict(cpu) {}
};
struct A3_base_2
{
	~A3_base_2() restrict(cpu,amp) {}
};
class A3 : public A3_base_1, public A3_base_2
{
	// defaulted: ~A3() restrict(cpu)
};

// Class with non-amp-compatible scalar data members
struct A4
{
	int i;
	char c;
	// defaulted: ~A4() restrict(cpu)
};

// Class with data member having defaulted dtors
union A5_member_1
{
	char c;
};
struct A5_member_2 {};
class A5
{
	A5_member_1 m1;
	A5_member_2 m2;
	// defaulted: ~A5() restrict(cpu)
};

// Class with data member having user-defined dtor
class A6_member_1 {};
struct A6_member_2
{
	~A6_member_2() restrict(cpu) {}
};
class A6
{
	A6_member_1 m1;
	A6_member_2 m2;
	// defaulted: ~A6() restrict(cpu)
};

// Classes with base classes having both defaulted and user-defined dtors
// and data members having both defaulted and user-defined dtors
class A7_base_1
{
	char c;
};
struct A7_base_2
{
	~A7_base_2() restrict(cpu,amp) {}
};
class A7_member_1 {};
struct A7_member_2
{
	~A7_member_2() restrict(cpu,amp) {}
};
class A7 : protected A7_base_1, public A7_base_2
{
	A7_member_1 m1;
	A7_member_2 m2;
	// defaulted: ~A7() restrict(cpu)
};

class A8_base_1 {};
struct A8_base_2
{
	~A8_base_2() restrict(cpu) {}
};
class A8_member_1 {};
struct A8_member_2
{
	~A8_member_2() restrict(cpu,amp) {}
};
class A8 : protected A8_base_1, public A8_base_2
{
	A8_member_1 m1;
	A8_member_2 m2;
	// defaulted: ~A8() restrict(cpu)
};

class A9_base_1 {};
struct A9_base_2
{
	~A9_base_2() restrict(cpu,amp) {}
};
struct A9_member_1
{
	~A9_member_1() restrict(cpu,amp) {}
};
class A9 : protected A9_base_1, public A9_base_2
{
	A9_member_1 m1;
	char c;
	// defaulted: ~A9() restrict(cpu)
};

class A10_base_1 {};
struct A10_base_2
{
	~A10_base_2() restrict(cpu,amp) {}
};
class A10_member_1 {};
struct A10_member_2
{
	~A10_member_2() restrict(cpu) {}
};
class A10 : protected A10_base_1, public A10_base_2
{
	A10_member_1 m1;
	A10_member_2 m2;
	// defaulted: ~A10() restrict(cpu)
};
