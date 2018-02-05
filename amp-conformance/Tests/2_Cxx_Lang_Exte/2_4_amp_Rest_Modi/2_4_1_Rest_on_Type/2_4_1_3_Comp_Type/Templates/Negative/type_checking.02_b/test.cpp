// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Validate type checking for class template members.</summary>

#include "../../type_checking_common.h"

#pragma warning(disable: 4101) // unreferenced local variable

template <int>
void f_3(obj_N<2>* p) restrict(AMP_RESTRICTION)
{
	p->i;
}

template <typename T>
void f_4(obj_N_T<7, T>* p) restrict(AMP_RESTRICTION)
{
	(*p).i;
}

template <int>
void f_6(derived_N_T<17, char>* p) restrict(AMP_RESTRICTION)
{
	(*p).i;
}

template <typename>
void f_8(member_N<27>* p) restrict(AMP_RESTRICTION)
{
	p->m;
}
//#Expects: Error: test\.cpp\(16\) : .+ C3581
//#Expects: Error: test\.cpp\(22\) : .+ C3581
//#Expects: Error: test\.cpp\(28\) : .+ C3581
//#Expects: Error: test\.cpp\(34\) : .+ C3581

// Note: restrict(amp) is intended on following functions, as they are explicitely instantiated
// and (cpu,amp) restricted are instatiated only for cpu context in such case.

template <typename>
void f_9(obj_N<3>& r) restrict(amp)
{
	r.i;
}

template <int>
void f_10(obj_N_T<8, char>& r) restrict(amp)
{
	r.i;
}

template <typename T>
void f_12(derived_N_T<18, T>& r) restrict(amp)
{
	r.i;
}

template <int>
void f_14(member_N<28>& r) restrict(amp)
{
	r.m;
}
//#Expects: Error: test\.cpp\(47\) : .+ C3581
//#Expects: Error: test\.cpp\(53\) : .+ C3581
//#Expects: Error: test\.cpp\(59\) : .+ C3581
//#Expects: Error: test\.cpp\(65\) : .+ C3581

void instantiate_templates() restrict(AMP_RESTRICTION)
{
	f_3<3>(nullptr);
	f_4<char>(nullptr);
	f_6<6>(nullptr);
	f_8<int>(nullptr);
}
template void f_9<int>(obj_N<3>& r) restrict(amp);
template void f_10<10>(obj_N_T<8, char>& r) restrict(amp);
template void f_12<char>(derived_N_T<18, char>& r) restrict(amp);
template void f_14<14>(member_N<28>& r) restrict(amp);

