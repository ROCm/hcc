// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Validate type checking for class template members.</summary>
// Note: this is a regression test for #396238.
#include "../../type_checking_common.h"

void f_1() restrict(AMP_RESTRICTION)
{
	obj_N<0> o1;
	obj_N_T<5, char> o2;
	derived_10_char o3;
	derived_N_T<15, char> o4;
	member_20 o5;
	member_N<25> o6;
}
//#Expects: Error: test\.cpp\(13\) : .+ C3581
//#Expects: Error: test\.cpp\(14\) : .+ C3581
//#Expects: Error: test\.cpp\(15\) : .+ C3581
//#Expects: Error: test\.cpp\(16\) : .+ C3581
//#Expects: Error: test\.cpp\(17\) : .+ C3581
//#Expects: Error: test\.cpp\(18\) : .+ C3581

void f_2() restrict(AMP_RESTRICTION)
{
	obj_N<1>();
	obj_N_T<6, char>();
	derived_11_char();
	derived_N_T<16, char>();
	member_21();
	member_N<26>();
}
//#Expects: Error: test\.cpp\(29\) : .+ C3581
//#Expects: Error: test\.cpp\(30\) : .+ C3581
//#Expects: Error: test\.cpp\(31\) : .+ C3581
//#Expects: Error: test\.cpp\(32\) : .+ C3581
//#Expects: Error: test\.cpp\(33\) : .+ C3581
//#Expects: Error: test\.cpp\(34\) : .+ C3581

void f_3(obj_N<2>* p) restrict(AMP_RESTRICTION)
{
	p->i;
}

void f_4(obj_N_T<7, char>* p) restrict(AMP_RESTRICTION)
{
	(*p).i;
}

void f_5(derived_12_char* p) restrict(AMP_RESTRICTION)
{}

void f_6(derived_N_T<17, char>* p) restrict(AMP_RESTRICTION)
{
	(*p).i;
}

void f_7(member_22* p) restrict(AMP_RESTRICTION)
{}

void f_8(member_N<27>* p) restrict(AMP_RESTRICTION)
{
	p->m;
}
//#Expects: Error: test\.cpp\(45\) : .+ C3581
//#Expects: Error: test\.cpp\(50\) : .+ C3581
//#Expects: Error: test\.cpp\(54\) : .+ C3581
//#Expects: Error: test\.cpp\(58\) : .+ C3581
//#Expects: Error: test\.cpp\(62\) : .+ C3581
//#Expects: Error: test\.cpp\(66\) : .+ C3581

void f_9(obj_N<3>& r) restrict(AMP_RESTRICTION)
{
	r.i;
}

void f_10(obj_N_T<8, char>& r) restrict(AMP_RESTRICTION)
{
	r.i;
}

void f_11(derived_13_char& r) restrict(AMP_RESTRICTION)
{}

void f_12(derived_N_T<18, char>& r) restrict(AMP_RESTRICTION)
{
	r.i;
}

void f_13(member_23& r) restrict(AMP_RESTRICTION)
{}

void f_14(member_N<28>& r) restrict(AMP_RESTRICTION)
{
	r.m;
}
//#Expects: Error: test\.cpp\(77\) : .+ C3581
//#Expects: Error: test\.cpp\(82\) : .+ C3581
//#Expects: Error: test\.cpp\(86\) : .+ C3581
//#Expects: Error: test\.cpp\(90\) : .+ C3581
//#Expects: Error: test\.cpp\(94\) : .+ C3581
//#Expects: Error: test\.cpp\(98\) : .+ C3581

void f_15(obj_N<4> o) restrict(AMP_RESTRICTION)
{}

void f_16(obj_N_T<9, char> o) restrict(AMP_RESTRICTION)
{}

void f_17(derived_14_char o) restrict(AMP_RESTRICTION)
{}

void f_18(derived_N_T<19, char> o) restrict(AMP_RESTRICTION)
{}

void f_19(member_24 o) restrict(AMP_RESTRICTION)
{}

void f_20(member_N<29> o) restrict(AMP_RESTRICTION)
{}
//#Expects: Error: test\.cpp\(108\) : .+ C3581
//#Expects: Error: test\.cpp\(111\) : .+ C3581
//#Expects: Error: test\.cpp\(114\) : .+ C3581
//#Expects: Error: test\.cpp\(117\) : .+ C3581
//#Expects: Error: test\.cpp\(120\) : .+ C3581
//#Expects: Error: test\.cpp\(123\) : .+ C3581

