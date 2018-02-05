// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test that destructor is called in the amp-restricted context.</summary>

#include <amptest.h>
#include <amptest_main.h>
using namespace concurrency;
using namespace concurrency::Test;

class obj
{
	array_view<int> counter;
public:
	obj(const array_view<int>& counter) restrict(cpu,amp) : counter(counter) {}
	obj(const obj& rhs) restrict(cpu,amp) : counter(rhs.counter) {}
	~obj() restrict(cpu,amp) { counter[0]++; }
};

runall_result test_main()
{
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	runall_result result;
	int counter_, amp_result_;
	array_view<int> counter(1, &counter_);
	array_view<int> amp_result(1, &amp_result_);

	// Object declared in the top-most scope.
	counter[0] = 0;
	parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp)
	{
		obj o(counter);
	});
	result &= REPORT_RESULT(counter[0] == 1);

	// Objects declared in local scopes, ensure the destructor is called in the correct place.
	counter[0] = 0;
	amp_result[0] = 0;
	parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp)
	{
		{
			obj o(counter);
			amp_result[0] |= counter[0] == 0 ? 0x1 : 0x0; // dtor not called yet
		}
		amp_result[0] |= counter[0] == 1 ? 0x2 : 0x0; // dtor called already

		{
			obj o(counter);
		}
		amp_result[0] |= counter[0] == 2 ? 0x4 : 0x0; // dtor called for the second object
	});
	result &= REPORT_RESULT(counter[0] == 2);
	result &= REPORT_RESULT(amp_result[0] == 0x7);

	// Object created in the for loop expression.
	counter[0] = 0;
	parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp)
	{
		for(int i = 0; i < 100; i++, obj(counter));
	});
	result &= REPORT_RESULT(counter[0] == 100);


	return result;
}
