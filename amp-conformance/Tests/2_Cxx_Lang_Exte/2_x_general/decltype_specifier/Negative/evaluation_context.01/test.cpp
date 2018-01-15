// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify that decltype evaluation is performed in the correct context, negative counterpart.</summary>
//#Expects: Error: test\.cpp\(28\) : .+ C3930:.*(\bb_1\b)
//#Expects: Error: test\.cpp\(29\) : .+ C3930:.*(\bb_2\b)
//#Expects: Error: test\.cpp\(37\) : .+ C3930:.*(\bb_1\b)
//#Expects: Error: test\.cpp\(44\) : .+ C3930:.*(\ba_1\b)
//#Expects: Error: test\.cpp\(45\) : .+ C3930:.*(\ba_2\b)
//#Expects: Error: test\.cpp\(53\) : .+ C3930:.*(\ba_1\b)
//#Expects: Error: test\.cpp\(60\) : .+ C2785:.*(\bfloat c_1\(void\)).*(\bA c_1\(void\) restrict\(amp\))

class A {};

int		a_1() restrict(cpu);
char	a_2() restrict(cpu);
int		b_1() restrict(amp);
float	b_2() restrict(amp);
float	c_1() restrict(cpu);
A		c_1() restrict(amp);

void f_cpu() restrict(cpu)
{
	// amp -> cpu
	decltype(b_1()); //error
	decltype(b_2()); //error
}

void f_cpu_nested() restrict(amp)
{
	[]() restrict(cpu)
	{
		// amp -> cpu
		decltype(b_1()); //error
	};
}

void f_amp() restrict(amp)
{
	// cpu -> amp
	decltype(a_1()); //error
	decltype(a_2()); //error
}

void f_amp_nested() restrict(cpu)
{
	[]() restrict(amp)
	{
		// cpu -> amp
		decltype(a_1()); //error
	};
}

void f_cpu_amp() restrict(cpu,amp)
{
	// cpu|amp (distinct) -> cpu,amp
	decltype(c_1()); //error
}

