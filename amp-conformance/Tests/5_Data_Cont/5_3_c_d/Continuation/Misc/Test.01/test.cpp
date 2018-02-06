// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>This test tests the continuation to depth 6. See comments for full description.</summary>

// ------------------------------
// This test tests the continuation to depth 6. The continuation passed to copy_async(...).then(...) function
// again invokes copy with destination of previous copy_async(...) as its source container. Looking at the code
// will give clearer idea of flow. Following is the sequence of copy done in test:
// 1) Iterator to array
// 2) Array to array_view
// 3) Array_view to array
// 4) Array to staging array
// 5) Staging array to array_view
// 6) Array_view to iterator
// ------------------------------

#include <amptest.h>
#include <amptest_main.h>
#include <amptest/event.h>

using namespace Concurrency;
using namespace Concurrency::Test;

#define DATA_SIZE 1024

runall_result test_main()
{
	accelerator_view cp_av = accelerator(accelerator::cpu_accelerator).get_default_view();
	accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

	std::vector<int> src_vec(DATA_SIZE), dest_vec(DATA_SIZE);
	Fill(src_vec);
	Concurrency::array<int, 1> arr1(DATA_SIZE, av);

	// Wait event is set when continuation finishes verification.
	// Each continuation waits for its child continuation to complete
	// because child continuation captures variable from enclosing scope.
	// If the continuation does not wait for child continuation to complete
	// it may happen that parent continuation finishes before child
	// continuation and the captured variables by child continuation will
	// go out of scope and no longer valid.
	event waitEvent1;
	completion_future cf = copy_async(src_vec.begin(), src_vec.end(), arr1);

	auto t1 = cf.to_task().then([&]() {
		Log(LogType::Info, true) << "Inside first continuation" << std::endl;
		std::vector<int> data1(DATA_SIZE);
		array_view<int, 1> arr_v1(DATA_SIZE, data1);

		event waitEvent2;
		copy_async(arr1, arr_v1).then([&]() {
			Log(LogType::Info, true) << "Inside second continuation" << std::endl;
			array<int, 1> arr2(DATA_SIZE, av);

			event waitEvent3;
			copy_async(arr_v1, arr2).then([&]() {
				Log(LogType::Info, true) << "Inside third continuation" << std::endl;
				array<int, 1> s_arr1(DATA_SIZE, cp_av, av);

				event waitEvent4;
				copy_async(arr2, s_arr1).then([&]() {
					Log(LogType::Info, true) << "Inside fourth continuation" << std::endl;
					std::vector<int> data2(DATA_SIZE);
					array_view<int, 1> arr_v2(DATA_SIZE, data2);

					event waitEvent5;
					copy_async(s_arr1, arr_v2).then([&]() {
						Log(LogType::Info, true) << "Inside fifth continuation" << std::endl;

						event waitEvent6;
						copy_async(arr_v2, dest_vec.begin()).then([&]() {
							Log(LogType::Info, true) << "Inside sixth continuation" << std::endl;
							waitEvent6.set();
						});

					waitEvent6.wait();
					waitEvent5.set();
					});
				waitEvent5.wait();
				waitEvent4.set();
				});
			waitEvent4.wait();
			waitEvent3.set();
			});
		waitEvent3.wait();
		waitEvent2.set();
		});
	waitEvent2.wait();
	waitEvent1.set();
	});

	waitEvent1.wait();

	return REPORT_RESULT(Verify(src_vec, dest_vec));
}

