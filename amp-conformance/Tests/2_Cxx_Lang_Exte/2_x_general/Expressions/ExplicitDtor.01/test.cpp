// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Explicit destructor call is allowed in amp context</summary>
#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

class obj
{
public:
	obj(const array_view<int>& scratch, int val) restrict(cpu,amp)
		: scratch(scratch)
		, val(val)
	{}

	~obj() restrict(cpu,amp)
	{
		scratch[0] += val;
	}

private:
	array_view<int> scratch;
	int val;
};

typedef obj obj_t;

template <typename T>
void destroy(const array_view<int>& res, const array_view<int>& scratch) restrict(amp)
{
	T t(scratch, 8);
	t.~T();
	if(scratch[0] & 8)
	{
		res[0] += 8;
	}
}

// In the following test obj destructor is called explicitely
// and immediately after it, the target data structure is
// checked for the expected result and success but is set on
// the final result, passed later to CPU context.
runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	int res_ = 0, scratch_ = 0;
	array_view<int> res(1, &res_);
	array_view<int> scratch(1, &scratch_);
	parallel_for_each(av, res.get_extent(), [=](index<1>) restrict(amp)
	{
		// "Straightforward" call
		obj o_1(scratch, 1);
		o_1.~obj();
		if(scratch[0] & 1)
		{
			res[0] += 1;
		}

		// Typedef call
		obj_t o_2(scratch, 2);
		o_2.~obj_t();
		if(scratch[0] & 2)
		{
			res[0] += 2;
		}

		// Pointer call
		obj o_3(scratch, 4);
		obj* o_ptr = &o_3;
		o_ptr->~obj();
		if(scratch[0] & 4)
		{
			res[0] += 4;
		}

		// Template call
		destroy<obj>(res, scratch);
	});

	return REPORT_RESULT(res[0] == 15);
}
