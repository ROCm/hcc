// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify that decltype evaluation is performed in the correct context.</summary>
#include <type_traits>
#define TEST(a,b) static_assert(std::is_same<a,b>::value, "Test failed, type of \"" #a "\" != type of \"" #b "\".")

class A {};

int		a_1() restrict(cpu);
char	a_2() restrict(cpu);
int		b_1() restrict(amp);
float	b_2() restrict(amp);
A		c_1() restrict(cpu,amp);
float	c_2() restrict(cpu,amp);
int		d_1() restrict(cpu);
int		d_1() restrict(amp);
float	e_1() restrict(cpu);
A		e_1() restrict(amp);

void f_cpu() restrict(cpu)
{
	// cpu -> cpu
	TEST(decltype(a_1()), int);
	TEST(decltype(a_2()), char);

	// cpu,amp -> cpu
	TEST(decltype(c_1()), A);
	TEST(decltype(c_2()), float);

	// cpu|amp -> cpu
	TEST(decltype(d_1()), int);

	// cpu|amp (distinct) -> cpu
	TEST(decltype(e_1()), float);
}

void f_cpu_nested() restrict(amp)
{
	[]() restrict(cpu)
	{
		// cpu|amp (distinct) -> cpu
		TEST(decltype(e_1()), float);
	};
}

void f_amp() restrict(amp)
{
	// amp -> amp
	TEST(decltype(b_1()), int);
	TEST(decltype(b_2()), float);

	// cpu,amp -> amp
	TEST(decltype(c_1()), A);
	TEST(decltype(c_2()), float);

	// cpu|amp -> amp
	TEST(decltype(d_1()), int);

	// cpu|amp (distinct) -> amp
	TEST(decltype(e_1()), A);
}

void f_amp_nested() restrict(cpu)
{
	[]() restrict(amp)
	{
		// cpu|amp (distinct) -> amp
		TEST(decltype(e_1()), A);
	};
}

void f_cpu_amp() restrict(cpu,amp)
{
	// cpu,amp -> cpu,amp
	TEST(decltype(c_1()), A);
	TEST(decltype(c_2()), float);

	// cpu|amp -> cpu,amp
	TEST(decltype(d_1()), int);
}
