// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using pointer to an array that is retrieved within the lambda is ok.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main() {
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	// Setup input data
	int val = 1;
	array<int,1> ary(1, &val, av);

	// Setup output
	int expected_new_val = 2012;
	parallel_for_each(av, ary.get_extent(), [=,&ary](index<1> idx) restrict(amp) {
		// Get a pointer to our input container
		// This line verifies the expected type of the captured array, so no casting is neccessary
		array<int,1>* ary_ptr = &ary;

		(*ary_ptr)[idx] = expected_new_val;
	});

	// Copy the results to the CPU and verify
	return VerifyAllSameValue(ary, expected_new_val) == -1;
}

