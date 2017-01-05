// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Pseudo destructor call is allowed in amp context</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

typedef int I;
typedef bool B;

template <typename T>
void destroy() restrict(amp)
{
	T t;
	t.~T();
}

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	int res_ = 0;
	array_view<int> res(1, &res_);
	parallel_for_each(av, res.get_extent(), [=](index<1>) restrict(amp)
	{
		int i = 0;
		i.~I();

		B b = false;
		b.B::~B();

		int* pi = &i;
		pi->~I(); // Note: should be pf->~decltype(*pi)(); but there's FE bug #384102

		destroy<long>();
	});

	return runall_pass; // Compile-time tests.
}
