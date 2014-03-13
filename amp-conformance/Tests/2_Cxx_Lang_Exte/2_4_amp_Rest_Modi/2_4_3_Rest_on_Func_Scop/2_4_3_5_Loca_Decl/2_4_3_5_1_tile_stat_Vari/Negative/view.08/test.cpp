// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>tile_static definition with array_view or writeonly_texture_view, base class + member field + union + array</summary>
//#Expects: Error: test\.cpp\(70\) : .+ C3584:.*ts_a
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

class J : public array_view<int>
{
};

class I
{
	const J j;
};

struct H
{
	const I i[3][2][1];
};

class G
{
	const H h[3];
};

struct F : protected G
{
};

class E
{
	F f;
};

struct D
{
public:
	E e[4][3][2][1];
};

class C
{
	union
	{
		struct
		{
			D d;
		};
	} u;
};

struct B
{
	C c[2];
};

class A : private B
{
};

void f() restrict(amp)
{
	tile_static A ts_a;
}

runall_result test_main()
{
	return runall_fail; // Should have not compiled.
}

