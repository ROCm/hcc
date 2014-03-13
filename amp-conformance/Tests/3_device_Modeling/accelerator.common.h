// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

// Test helper API's

#pragma once

#include <amptest.h>

namespace
{

    // Compares properties of two accelerators.
	bool is_accelerator_equal(const concurrency::accelerator& acc1, const concurrency::accelerator& acc2)
	{

        // Not using operator== on purpose.
		// Assuming that equality of all properties implies the same underlying device.
		return acc1.get_device_path() == acc2.get_device_path()
			&& acc1.get_version() == acc2.get_version()
			&& acc1.get_description() == acc2.get_description()
			&& acc1.get_is_debug() == acc2.get_is_debug()
			&& acc1.get_is_emulated() == acc2.get_is_emulated()
			&& acc1.get_has_display() == acc2.get_has_display()
			&& acc1.get_supports_double_precision() == acc2.get_supports_double_precision()
			&& acc1.get_supports_limited_double_precision() == acc2.get_supports_limited_double_precision()
			&& acc1.get_dedicated_memory() == acc2.get_dedicated_memory();
	}

	// Compares properties of two accelerator views.
	bool is_accelerator_view_equal(const concurrency::accelerator_view& av1, const concurrency::accelerator_view& av2)
	{

        // Not using operator== on purpose.
		// Assuming that equality of all properties implies the same underlying device.
		return av1.get_accelerator() == av2.get_accelerator()
			&& av1.get_is_debug() == av2.get_is_debug()
			&& av1.get_version() == av2.get_version()
			&& av1.get_queuing_mode() == av2.get_queuing_mode();
	}

	// Runs a copy algorithm on a given accelerator_view.
	bool is_accelerator_view_operable(const concurrency::accelerator_view& av)
	{

        const size_t DATA_SIZE = 1024;
		std::vector<int> inData(DATA_SIZE);
		std::vector<int> outData(DATA_SIZE);
		std::mt19937 rng;
		std::generate(inData.begin(), inData.end(), rng);

		concurrency::array<int, 1> arr(concurrency::extent<1>(DATA_SIZE), inData.begin(), av);
		concurrency::copy(arr, outData.begin());

		return std::equal(inData.begin(), inData.end(), outData.begin());
	}

	// Exits when the default device is not available in the system.
	bool run_simple_p_f_e(const concurrency::accelerator_view& av)
	{

		std::vector<int> input(16);
		std::vector<int> output(input.size());
		concurrency::array_view<int, 1> input_view(static_cast<int>(input.size()), input);
		concurrency::array_view<int, 1> output_view(static_cast<int>(output.size()), output);

		std::mt19937 rng;
		std::generate(input.begin(), input.end(), rng);

		concurrency::parallel_for_each(av, input_view.get_extent(),
			[=](concurrency::index<1> idx) restrict(amp)
			{
				output_view[idx] = input_view[idx] / 2;
			}
		);
		output_view.synchronize();

		auto pair = std::mismatch(input.begin(), input.end(), output.begin(),
			[](int lhs, int rhs) -> bool
			{
				return rhs == lhs / 2;
			}
		);

		return pair.first == input.end()
			&& pair.second == output.end();
	}
}
