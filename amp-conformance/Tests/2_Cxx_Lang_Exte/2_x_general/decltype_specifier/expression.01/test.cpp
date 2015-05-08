// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify decltype expression evaluation.</summary>
#include <type_traits>
#include <amp.h>
#define TEST(a,b) static_assert(std::is_same<a,b>::value, "Test failed, type of \"" #a "\" != type of \"" #b "\".")
#pragma warning(disable: 4101 4189) // unreferenced local variable; unreferenced initialized local variable

struct struct_A {};

// Functions
bool	f_a_1() restrict(cpu);
char	f_a_2();
int		f_b_1() restrict(amp);
float*	f_c_1(int) restrict(cpu,amp);
int		f_c_2() restrict(cpu);
float	f_c_2() restrict(amp);
int		f_d_1(int) restrict(cpu);
float	f_d_1(float) restrict(amp);
int		f_e_1() restrict(cpu);
float	f_e_1(int = 0) restrict(amp);
float	f_f_1(int, ...) restrict(cpu);
bool	f_f_1(int) restrict(amp);

// Templated functions
template <typename T> T   tf_a_1(T) restrict(cpu,amp);
template <typename T> T   tf_b_1(T) restrict(cpu);
template <typename T> T   tf_b_1(T) restrict(amp);
template <typename T> T   tf_c_1(T) restrict(cpu);
template <> float		  tf_c_1<float>(float) restrict(cpu);
int						  tf_c_1(int) restrict(amp);
float					  tf_c_1(float) restrict(amp);
template <typename T> T   tf_d_1(T) restrict(cpu);
template <typename T> int tf_d_1(T) restrict(amp);

// Template classes
template <typename T = char>
struct obj_T
{};

// Argument-dependent lookup
namespace namespace_A
{
	struct struct_B {};
	int f(struct_B) restrict(cpu);
}
int f(namespace_A::struct_B) restrict(amp);

// Tests with result specific to cpu context
void f_cpu() restrict(cpu)
{
	// Tests for expressions within a block statement
	{
		// Argument conversions
		float f;
#pragma warning(push)
#pragma warning(disable: 4244) // conversion from 'float' to 'int'
		TEST(decltype(f_d_1(1.f)), int); // implicit floating-integral conversion of the argument
		TEST(decltype(f_d_1(f)), int);
#pragma warning(pop)

		// Various number of arguments
		TEST(decltype(f_e_1()), int);
		TEST(decltype(f_f_1(1)), float);
		TEST(decltype(f_f_1(1, 2)), float);

		// Return type
		TEST(decltype(tf_d_1('x')), char);
	}

	// Repeat tests for expression within a lambda's compound statement
	[]
	{
		// Argument conversions
		float f;
#pragma warning(push)
#pragma warning(disable: 4244) // conversion from 'float' to 'int'
		TEST(decltype(f_d_1(1.f)), int); // implicit floating-integral conversion of the argument
		TEST(decltype(f_d_1(f)), int);
#pragma warning(pop)

		// Various number of arguments
		TEST(decltype(f_e_1()), int);
		TEST(decltype(f_f_1(1)), float);
		TEST(decltype(f_f_1(1, 2)), float);

		// Return type
		TEST(decltype(tf_d_1('x')), char);
	};
}

// Tests with result specific to amp context
void f_amp() restrict(amp)
{
	// Tests for expressions within a block statement
	{
	// Casts
		TEST(decltype(f_b_1()), int);
		TEST(decltype((float)f_b_1()), float);
		TEST(decltype(static_cast<float>(f_b_1())), float);

		// Argument conversions
		int i;
#pragma warning(push)
#pragma warning(disable: 4244) // conversion from 'int' to 'float'
		TEST(decltype(f_d_1(1)), float); // implicit floating-integral conversion of the argument
		TEST(decltype(f_d_1(i)), float);
#pragma warning(pop)

		// Various number of arguments
		TEST(decltype(f_e_1()), float);
		TEST(decltype(f_f_1(1)), bool);

		// Return type
		TEST(decltype(tf_d_1(2.0)), int);

		// tile_static variables
		tile_static float f;
		tile_static struct_A pca;
		tile_static int ia[5][4][3];
		TEST(decltype(f), float);
		TEST(decltype((f)), float&);
		TEST(decltype(pca), struct_A);
		TEST(decltype((pca)), struct_A&);
		TEST(decltype(ia), int[5][4][3]);
		TEST(decltype((ia)), int(&)[5][4][3]);
	}

	// Repeat tests for expression within a lambda's compound statement
	[]
	{
		// Casts
		TEST(decltype(f_b_1()), int);
		TEST(decltype((float)f_b_1()), float);
		TEST(decltype(static_cast<float>(f_b_1())), float);

		// Argument conversions
		int i;
	#pragma warning(push)
	#pragma warning(disable: 4244) // conversion from 'int' to 'float'
		TEST(decltype(f_d_1(1)), float); // implicit floating-integral conversion of the argument
		TEST(decltype(f_d_1(i)), float);
	#pragma warning(pop)

		// Various number of arguments
		TEST(decltype(f_e_1()), float);
		TEST(decltype(f_f_1(1)), bool);

		// Return type
		TEST(decltype(tf_d_1(2.0)), int);

		// tile_static variables
		tile_static float f;
		tile_static struct_A pca;
		tile_static int ia[5][4][3];
		TEST(decltype(f), float);
		TEST(decltype((f)), float&);
		TEST(decltype(pca), struct_A);
		TEST(decltype((pca)), struct_A&);
		TEST(decltype(ia), int[5][4][3]);
		TEST(decltype((ia)), int(&)[5][4][3]);
	};
}

// Tests with the same result is cpu and amp contexts
void f_cpu_amp() restrict(cpu,amp)
{
	// Tests for expressions within a block statement
	{
		struct_A object_A;

		// Function type
		TEST(decltype(f_a_1), bool() restrict(cpu));
		TEST(decltype(f_a_2), char() restrict(cpu));
		TEST(decltype(f_b_1), int() restrict(amp));
		TEST(decltype(f_c_1), float*(int) /*restrict(cpu,amp)*/);

		// Return type
		TEST(decltype(f_c_1(1)), float*);
		TEST(decltype(tf_d_1(1)), int);

		// Templated function type
		TEST(decltype(tf_a_1<double>), double(double));
		TEST(decltype(tf_a_1<const int&>), const int&(const int&));

		// Return type from templated function
		TEST(decltype(tf_a_1(1)), int);
		TEST(decltype(tf_a_1(false)), bool);
		TEST(decltype(tf_a_1(struct_A())), struct_A);
		TEST(decltype(tf_b_1(&object_A)), struct_A*);
		TEST(decltype(tf_c_1(1)), int);
		TEST(decltype(tf_c_1(1.f)), float);

		// Return type from ADL function
		/* // FE bug #386834
		TEST(decltype(f(namespace_A::struct_B())), int); */

		// Comma operator
		TEST(decltype((1,2.f,3u)), unsigned);

		// Use of literals of amp-unsupported type
		TEST(decltype('a' ? 1 : 1), int);
		TEST(decltype(true ? 'a' : 1), int);
		TEST(decltype(1, 'a', 2), int);

		// Use of amp-unsupported type as template argument
		TEST(decltype(obj_T<char>()), obj_T<char>);
		TEST(decltype(obj_T<>()), obj_T<char>);

		// Variables
		float f;
		const bool& rb = false;
		const struct_A* pca;
		int ia[5][4][3];
		TEST(decltype(f), float);
		TEST(decltype((f)), float&);
		TEST(decltype(rb), const bool&);
		TEST(decltype((rb)), const bool&);
		TEST(decltype(pca), const struct_A*);
		TEST(decltype((pca)), const struct_A*&);
		TEST(decltype(ia), int[5][4][3]);
		TEST(decltype((ia)), int(&)[5][4][3]);
		[=]
		{
			TEST(decltype(f), float);
			TEST(decltype((f)), const float&);
			//TEST(decltype(rb), const bool&); // FE bug #386754
			TEST(decltype((rb)), const bool&);
		};
	}

	// Repeat tests for expression within a lambda's compound statement
	[]
	{
		struct_A object_A;

		// Function type
		TEST(decltype(f_a_1), bool() restrict(cpu));
		TEST(decltype(f_a_2), char() restrict(cpu));
		TEST(decltype(f_b_1), int() restrict(amp));
		TEST(decltype(f_c_1), float*(int) /*restrict(cpu,amp) is implicit (from the current function context)*/);

		// Return type
		TEST(decltype(f_c_1(1)), float*);
		TEST(decltype(tf_d_1(1)), int);

		// Templated function type
		TEST(decltype(tf_a_1<double>), double(double));
		TEST(decltype(tf_a_1<const int&>), const int&(const int&));

		// Return type from templated function
		TEST(decltype(tf_a_1(1)), int);
		TEST(decltype(tf_a_1(false)), bool);
		TEST(decltype(tf_a_1(struct_A())), struct_A);
		TEST(decltype(tf_b_1(&object_A)), struct_A*);
		TEST(decltype(tf_c_1(1)), int);
		TEST(decltype(tf_c_1(1.f)), float);

		// Return type from ADL function
		/* // FE bug #386834
		TEST(decltype(f(namespace_A::struct_B())), int); */

		// Comma operator
		TEST(decltype((1,2.f,3u)), unsigned);

		// Use of literals of amp-unsupported type
		TEST(decltype('a' ? 1 : 1), int);
		TEST(decltype(true ? 'a' : 1), int);
		TEST(decltype(1, 'a', 2), int);

		// Use of amp-unsupported type as template argument
		TEST(decltype(obj_T<char>()), obj_T<char>);
		TEST(decltype(obj_T<>()), obj_T<char>);

		// Variables
		float f;
		const bool& rb = false;
		const struct_A* pca;
		int ia[5][4][3];
		TEST(decltype(f), float);
		TEST(decltype((f)), float&);
		TEST(decltype(rb), const bool&);
		TEST(decltype((rb)), const bool&);
		TEST(decltype(pca), const struct_A*);
		TEST(decltype((pca)), const struct_A*&);
		TEST(decltype(ia), int[5][4][3]);
		TEST(decltype((ia)), int(&)[5][4][3]);
		[=]
		{
			TEST(decltype(f), float);
			TEST(decltype((f)), const float&);
			// TEST(decltype(rb), const bool&); // FE bug #386754
			TEST(decltype((rb)), const bool&);
		};
	};
}

// Rudimentary tests for expressions in a trailing return type
auto f_trt_cpu_1(int i) -> decltype(f_d_1(i));
TEST(decltype(f_trt_cpu_1), int(int));

auto f_trt_cpu_2() -> decltype(f_f_1(1));
TEST(decltype(f_trt_cpu_2), float());

auto f_trt_cpu_3() -> decltype(tf_d_1(1ll));
TEST(decltype(f_trt_cpu_3), long long());

auto f_trt_amp_1() restrict(amp) -> decltype((float)f_b_1());
TEST(decltype(f_trt_amp_1), float() restrict(amp));

auto f_trt_amp_2() restrict(amp) -> decltype(f_d_1(1));
TEST(decltype(f_trt_amp_2), float() restrict(amp));

auto f_trt_amp_3() restrict(amp) -> decltype(tf_d_1(2.0));
TEST(decltype(f_trt_amp_3), int() restrict(amp));

auto f_trt_cpu_amp_1() restrict(cpu,amp) -> decltype(tf_a_1(false));
TEST(decltype(f_trt_cpu_amp_1), bool() restrict(cpu,amp));

auto f_trt_cpu_amp_2() restrict(cpu,amp) -> decltype(tf_c_1(1));
TEST(decltype(f_trt_cpu_amp_2), int() restrict(cpu,amp));

auto f_trt_cpu_amp_3() restrict(cpu,amp) -> decltype(tf_c_1(1.f));
TEST(decltype(f_trt_cpu_amp_3), float() restrict(cpu,amp));

auto f_trt_cpu_amp_4() restrict(cpu,amp) -> decltype('a', 1);
TEST(decltype(f_trt_cpu_amp_4), int() restrict(cpu,amp));

auto f_trt_cpu_amp_5() restrict(cpu,amp) -> decltype(obj_T<char>());
TEST(decltype(f_trt_cpu_amp_5), obj_T<char>() restrict(cpu,amp));
