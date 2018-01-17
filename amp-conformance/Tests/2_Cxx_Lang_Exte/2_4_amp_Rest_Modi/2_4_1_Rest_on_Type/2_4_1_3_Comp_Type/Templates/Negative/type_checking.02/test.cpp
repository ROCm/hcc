// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Validate type checking for class template members.</summary>
// Note: this is a regression test for #396238.
#include "../../type_checking_common.h"

#pragma warning(disable: 4101) // unreferenced local variable

template <typename T>
void f_1() restrict(AMP_RESTRICTION)
{
	obj_N<0> o1;
	obj_N_T<5, T> o2;
	derived_10_char o3;
	derived_N_T<15, T> o4;
	member_20 o5;
	member_N<25> o6;
}
//#Expects: Error: test\.cpp\(16\) : .+ C3581
//#Expects: Error: test\.cpp\(17\) : .+ C3581
//#Expects: Error: test\.cpp\(18\) : .+ C3581
//#Expects: Error: test\.cpp\(19\) : .+ C3581
//#Expects: Error: test\.cpp\(20\) : .+ C3581
//#Expects: Error: test\.cpp\(21\) : .+ C3581

template <int>
void f_2() restrict(AMP_RESTRICTION)
{
	obj_N<1>();
	obj_N_T<6, char>();
	derived_11_char();
	derived_N_T<16, char>();
	member_21();
	member_N<26>();
}
//#Expects: Error: test\.cpp\(33\) : .+ C3581
//#Expects: Error: test\.cpp\(34\) : .+ C3581
//#Expects: Error: test\.cpp\(35\) : .+ C3581
//#Expects: Error: test\.cpp\(36\) : .+ C3581
//#Expects: Error: test\.cpp\(37\) : .+ C3581
//#Expects: Error: test\.cpp\(38\) : .+ C3581

void instantiate_templates() restrict(AMP_RESTRICTION)
{
	f_1<char>();
	f_2<2>();
}

