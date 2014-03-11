// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create array with template element containing non-amp-restricted type</summary>
//#Expects: Error: test.cpp\(29\).?:.*(\bConcurrency::array_view<_Value_type>)
//#Expects: Error: test.cpp\(30\).?:.*(\bConcurrency::array_view<_Value_type>)
//#Expects: Error: test.cpp\(32\).?:.*(\bConcurrency::array<_Value_type>)
//#Expects: Error: test.cpp\(33\).?:.*(\bConcurrency::array<_Value_type>)

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::Test;

// Use N to force separate instantiation.
template <int N, typename T>
struct obj_T
{
	T m;
	int i;
};

void f() restrict(amp)
{
	array_view< obj_T<2, char> > av_1(0, nullptr);
	array_view< obj_T<2, array_view<int>> > av_2(0, nullptr);
	
	array< obj_T<3, char> > a_1(1);
	array< obj_T<3, array_view<int>> > a_2(1);
}

