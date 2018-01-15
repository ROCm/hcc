// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

// Empty class with user-defined dtor
class A1
{
public:
	~A1() restrict(cpu) {}
	// defaulted: A1() restrict(cpu)
};

// Empty class with base class having user-defined default ctor
struct A2_base
{
	A2_base() restrict(cpu) {}
};
class A2 : public A2_base
{
	// defaulted: A2() restrict(cpu)
};

// Empty class with two base classes having user-defined default ctors, one more restrictive than the other
struct A3_base_1
{
	int i;
	A3_base_1() restrict(cpu,amp) {}
};
class A3_base_2
{
	int i;
public:
	A3_base_2() restrict(cpu) {}
};
struct A3 : A3_base_1, A3_base_2
{
	// defaulted: A3() restrict(cpu)
};

// Class with data member having user-defined default ctor
class A4_member_1
{
public:
	A4_member_1() restrict(cpu) {}
};
class A4
{
	A4_member_1 m1;
	// defaulted: A4() restrict(cpu)
};

// Class with data members having user-defined default ctors, one more restrictive than the other
struct A5_member_1
{
	int i;
	A5_member_1() restrict(cpu,amp) {}
};
union A5_member_2
{
	int i;
	A5_member_2() restrict(cpu) {}
};
struct A5
{
	A5_member_1 m1;
	A5_member_2 m2;
	// defaulted: A5() restrict(cpu)
};
