// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Validate type checking for class template members.</summary>

#include "../../type_checking_common.h"

#pragma warning(disable: 4101) // unreferenced local variable


template <typename>
void f_5(derived_12_char* p) restrict(AMP_RESTRICTION)
{}

template <int>
void f_7(member_22* p) restrict(AMP_RESTRICTION)
{}

// Note: restrict(amp) is intended on following functions, as they are explicitely instantiated
// and (cpu,amp) restricted are instatiated only for cpu context in such case.

template <int>
void f_11(derived_13_char& r) restrict(amp)
{}

template <typename>
void f_13(member_23& r) restrict(amp)
{}

template <int>
void f_15(obj_N<4> o) restrict(amp)
{}

template <typename>
void f_16(obj_N_T<9, char> o) restrict(amp)
{}

template <typename>
void f_17(derived_14_char o) restrict(amp)
{}

template <int>
void f_18(derived_N_T<19, char> o) restrict(amp)
{}

template <int>
void f_19(member_24 o) restrict(amp)
{}

template <typename>
void f_20(member_N<29> o) restrict(amp)
{}

void instantiate_templates() restrict(AMP_RESTRICTION)
{
	f_5<int>(nullptr);
	f_7<7>(nullptr);
}
//#Expects: Error: test\.cpp\(59\) : .+ C3581
//#Expects: Error: test\.cpp\(60\) : .+ C3581

template void f_11<11>(derived_13_char& r) restrict(amp);
template void f_13<int>(member_23& r) restrict(amp);
//#Expects: Error: test\.cpp\(65\) : .+ C3581
//#Expects: Error: test\.cpp\(66\) : .+ C3581

template void f_15<15>(obj_N<4> o) restrict(amp);
template void f_16<int>(obj_N_T<9, char> o) restrict(amp);
template void f_17<int>(derived_14_char o) restrict(amp);
template void f_18<18>(derived_N_T<19, char> o) restrict(amp);
template void f_19<19>(member_24 o) restrict(amp);
template void f_20<int>(member_N<29> o) restrict(amp);
//#Expects: Error: test\.cpp\(70\) : .+ C3581
//#Expects: Error: test\.cpp\(71\) : .+ C3581
//#Expects: Error: test\.cpp\(72\) : .+ C3581
//#Expects: Error: test\.cpp\(73\) : .+ C3581
//#Expects: Error: test\.cpp\(74\) : .+ C3581
//#Expects: Error: test\.cpp\(75\) : .+ C3581

