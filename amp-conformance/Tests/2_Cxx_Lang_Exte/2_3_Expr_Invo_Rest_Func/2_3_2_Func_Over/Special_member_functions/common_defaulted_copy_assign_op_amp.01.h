// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

// Empty class with base class having user-defined copy op=
struct A1_base
{
	A1_base& operator=(const A1_base&) restrict(amp) { return *this; }
};
class A1 : public A1_base
{
	// defaulted: A1& operator=(const A1&) restrict(amp)
};

// Empty class with two base classes having user-defined copy op=, one more restrictive than the other
struct A2_base_1
{
	int i;
	A2_base_1& operator=(const A2_base_1&) restrict(cpu,amp) { return *this; }
};
class A2_base_2
{
	int i;
public:
	A2_base_2& operator=(const A2_base_2&) restrict(amp) { return *this; }
};
struct A2 : A2_base_1, A2_base_2
{
	// defaulted: A2& operator=(const A2&) restrict(amp)
};

// Class with data member having user-defined copy op=
class A3_member_1
{
public:
	A3_member_1& operator=(const A3_member_1&) restrict(amp) { return *this; }
};
class A3
{
	A3_member_1 m1;
	// defaulted: A3& operator=(const A3&) restrict(amp)
};

// Class with data members having user-defined copy op=, one more restrictive than the other
struct A4_member_1
{
	int i;
	A4_member_1& operator=(const A4_member_1&) restrict(cpu,amp) { return *this; }
};
union A4_member_2
{
	int i;
	A4_member_2& operator=(const A4_member_2&) restrict(amp) { return *this; }
};
struct A4
{
	A4_member_1 m1;
	A4_member_2 m2;
	// defaulted: A4& operator=(const A4&) restrict(amp)
};

// Classes with base classes and data members having user-defined copy op=
struct A5_base_1
{
	int i;
	A5_base_1& operator=(const A5_base_1&) restrict(amp) { return *this; }
};
struct A5_member_1
{
	int i;
	A5_member_1& operator=(const A5_member_1&) restrict(cpu,amp) { return *this; }
};
struct A5 : A5_base_1
{
	A5_member_1 m1;
	// defaulted: A5& operator=(const A5&) restrict(amp)
};

struct A6_base_1
{
	int i;
	A6_base_1& operator=(const A6_base_1&) restrict(cpu,amp) { return *this; }
};
struct A6_member_1
{
	int i;
	A6_member_1& operator=(const A6_member_1&) restrict(amp) { return *this; }
};
struct A6 : A6_base_1
{
	A6_member_1 m1;
	// defaulted: A6& operator=(const A6&) restrict(amp)
};
