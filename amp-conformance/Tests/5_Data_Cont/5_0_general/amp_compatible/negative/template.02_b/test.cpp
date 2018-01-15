// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create texture with template element containing non-amp-restricted type</summary>
//#Expects: Error: test.cpp\(28\).?:.*(\bConcurrency::graphics::texture<_Value_type,_Rank>)

#include <amp_graphics.h>
#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::graphics;
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
	texture<obj_T<0, array_view<int>>, 1> tex(1);
}

