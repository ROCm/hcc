// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

//
// Constructs required by tests
//
void	f_1() restrict(cpu);
void	f_1() restrict(amp);
int		f_2() restrict(cpu);
float	f_2() restrict(amp);
int		f_3() restrict(cpu);
float	f_4() restrict(amp);
int		f_5(int)   restrict(cpu);
int		f_5(float) restrict(amp);
template <typename T> bool tf_1(T) restrict(cpu,amp);
template <typename T> T    tf_2(T)     restrict(cpu);
                      int  tf_2(float) restrict(amp);

struct obj_conv_amp_cpu
{
	operator int() restrict(amp)
	{
		return 0;
	}
	operator float() restrict(cpu)
	{
		return 0;
	}
};

struct obj_conv_T
{
	template <typename T>
	operator T() restrict(cpu,amp);
};

template <typename T>
struct obj_T_m
{
	T m;
	int x;
};

template <typename T = char>
struct obj_T_m_def_char
{
	T m;
	int x;
};

template <int N>
struct obj_N
{};

char global_char;
int global_int;

//
// Tests prototypes
// TEST_CPU_* should fail in cpu-restricted context
// TEST_AMP_* should fail in amp-restricted context
// TEST_CPU_AMP_* should fail in cpu,amp-restricted context
//

// Incompatible restriction specifiers
#define TEST_CPU_1		decltype(f_4())


// amp-incompatible types of subexpressions
#define TEST_AMP_1		decltype(char() + 4)
#define TEST_AMP_2		decltype((char('x') + 4))
#define TEST_AMP_3		decltype(static_cast<int>(char('x')))
#define TEST_AMP_4		decltype((int)char('x'))
#define TEST_AMP_5		decltype(int(char('x')))
#define TEST_AMP_6		decltype(new int)
#define TEST_AMP_7		decltype(true ? 1.f : char('x'))
#define TEST_AMP_8		decltype(char() ? 1 : 1)
#define TEST_AMP_9		decltype(char() == 'a' ? 1 : 1)
#define TEST_AMP_10		decltype(true ? 1 : char())
#define TEST_AMP_11		decltype(true ? obj_conv_T() : 'a') // implicit obj -> char
#define TEST_AMP_12		decltype(1, char(), 2)
#define TEST_AMP_13		decltype(1, 2, 'a')
#define TEST_AMP_14		decltype(true || char())
#define TEST_AMP_15		decltype(false && char())
#define TEST_AMP_16		decltype(4294967296ll ? 1 : 1)
#define TEST_AMP_17		decltype(true ? (true ? 1 : char()) : 1)
#define TEST_AMP_18		decltype(obj_T_m<char>())
#define TEST_AMP_19		decltype(obj_T_m_def_char<>())
#define TEST_AMP_20		decltype(obj_N<(char())>())
#define TEST_AMP_21(_v)		decltype(_v = char())
#define TEST_AMP_22(_v)		decltype(_v += char())
#define TEST_AMP_23(_v)		decltype(_v | char())
#define TEST_AMP_24(_v)		decltype(_v ^ char())
#define TEST_AMP_25(_v)		decltype(_v & char())
#define TEST_AMP_26(_v)		decltype(_v == char())

// Access to global variable
#define TEST_AMP_27		decltype(global_char ? 1 : 1)
#define TEST_AMP_28		decltype(global_int);

// Throw expression
#define TEST_AMP_29		decltype(true ? 1 : throw 1)

// Incompatible restriction specifiers
#define TEST_AMP_30		decltype(f_3())


// Set of overloaded functions
#define TEST_CPU_AMP_1	decltype(f_1)
#define TEST_CPU_AMP_2	decltype(f_2)

// Different types of subexpressions
#define TEST_CPU_AMP_3	decltype(f_2())
#define TEST_CPU_AMP_4	decltype(static_cast<int>(f_2()))
#define TEST_CPU_AMP_5	decltype(obj_conv_amp_cpu() + 1) // different type of user defined conversion
#define TEST_CPU_AMP_6	decltype(f_5(f_2()))
#define TEST_CPU_AMP_7	decltype(tf_1(f_2()))
#define TEST_CPU_AMP_8	decltype(tf_2(1.f))

// Incompatible restriction specifiers
#define TEST_CPU_AMP_9	decltype(f_3());
#define TEST_CPU_AMP_10	decltype(f_4());
