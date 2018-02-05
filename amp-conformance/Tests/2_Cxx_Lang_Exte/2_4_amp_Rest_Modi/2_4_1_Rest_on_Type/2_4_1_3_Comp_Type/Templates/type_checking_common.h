// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

// This a common header used by type_checking_* tests (both positive and negative).
// The templates below use template parameter N to achieve separate
// template instantiations throughout the tests, e.g. obj_N<0> and obj_N<1>
// are instantiated separately and non-related in FE.

// Use N=0..4
template <int N>
struct obj_N
{
	char m;
	int i;
};

// Use N=5..9
template <int N, typename T>
struct obj_N_T
{
	T m;
	int i;
};

struct derived_10_char : obj_N_T<10, char> {};
struct derived_11_char : obj_N_T<11, char> {};
struct derived_12_char : obj_N_T<12, char> {};
struct derived_13_char : obj_N_T<13, char> {};
struct derived_14_char : obj_N_T<14, char> {};

// Use N=15..19
template <int N, typename T>
struct derived_N_T : obj_N_T<N, T>
{
};

struct member_20 { obj_N<20> m; };
struct member_21 { obj_N<21> m; };
struct member_22 { obj_N<22> m; };
struct member_23 { obj_N<23> m; };
struct member_24 { obj_N<24> m; };

// Use N=25..29
template <int N>
struct member_N
{
	obj_N<N> m;
};
