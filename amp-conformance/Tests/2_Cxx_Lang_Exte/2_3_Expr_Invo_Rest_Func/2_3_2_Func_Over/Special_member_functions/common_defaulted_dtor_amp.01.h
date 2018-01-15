// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

// Empty class with base class having user-defined dtor
struct A1_base
{
	~A1_base() restrict(amp) {}
};
struct A1 : public A1_base
{
	// defaulted: ~A1() restrict(amp)
};

// Empty class with base classes having both defaulted and user-defined dtors
class A2_base_1 { int i; };
class A2_base_2
{
	int i;
public:
	~A2_base_2() restrict(amp) {}
};
class A2 : A2_base_1, public A2_base_2
{
	// defaulted: ~A2() restrict(amp)
};

// Empty class with base classes having user-defined dtors, one more restrictive than the other
class A3_base_1
{
	int i;
public:
	~A3_base_1() restrict(amp) {}
};
struct A3_base_2
{
	int i;
	~A3_base_2() restrict(cpu,amp) {}
};
class A3 : public A3_base_1, public A3_base_2
{
	// defaulted: ~A3() restrict(amp)
};

// Class with data member having user-defined dtor
class A4_member_1 {};
struct A4_member_2
{
	int i;
	~A4_member_2() restrict(amp) {}
};
class A4
{
	A4_member_1 m1;
	A4_member_2 m2;
	// defaulted: ~A4() restrict(amp)
};

// Classes with base classes having both defaulted and user-defined dtors
// and data members having both defaulted and user-defined dtors
class A5_base_1 { int i; };
struct A5_base_2
{
	int i;
	~A5_base_2() restrict(amp) {}
};
class A5_member_1 { int i; };
struct A5_member_2
{
	int i;
	~A5_member_2() restrict(cpu,amp) {}
};
class A5 : protected A5_base_1, public A5_base_2
{
	A5_member_1 m1;
	A5_member_2 m2;
	// defaulted: ~A5() restrict(amp)
};

class A6_base_1 { int i; };
struct A6_base_2
{
	int i;
	~A6_base_2() restrict(cpu,amp) {}
};
class A6_member_1 { int i; };
struct A6_member_2
{
	int i;
	~A6_member_2() restrict(amp) {}
};
class A6 : protected A6_base_1, public A6_base_2
{
	A6_member_1 m1;
	A6_member_2 m2;
	// defaulted: ~A6() restrict(amp)
};
