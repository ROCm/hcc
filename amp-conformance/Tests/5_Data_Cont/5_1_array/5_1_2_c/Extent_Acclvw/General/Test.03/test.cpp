// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a array of nthe dimension with all extents 1</summary>

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
	vector<accelerator> devices = get_available_devices(device_flags::NOT_SPECIFIED);
	printf("Found %d devices\n", devices.size());

	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator device = devices[i];

		printf("Device %d: %ws (%ws)\n", i, device.get_description().c_str(), device.get_device_path().c_str());
		printf("Version %d \t Memory %u\n", device.get_version(), device.get_dedicated_memory());
		printf("Debug:%c \t Emulated:%c \t Display: %c\n", device.get_is_debug() ? 'Y' : 'N', device.get_is_emulated() ? 'Y' : 'N', device.get_has_display() ? 'Y' : 'N');

		{
			int edata[_rank];

			for (int i = 0; i < _rank; i++)
				edata[i] = 1;

			extent<_rank> e1(edata);
			array<_type, _rank> src(e1, device.get_default_view());

			// let the kernel initialize data;
			parallel_for_each(e1, [&](index<_rank> idx) __GPU_ONLY
			{
				src[idx] = _rank;
			});

			// Copy data to CPU
			vector<_type> opt(e1.size());
			opt = src;

			for (unsigned int i = 0; i < e1.size(); i++)
			{
				if (opt[i] != _rank)
					return false;
			}
		}

		printf("Finished with device %d\n", i);
	}

	return true;
}

runall_result test_main()
{
	runall_result result = false;

// XXX bypass the test and make it fail directly
#if 0
    result &= REPORT_RESULT((test_feature<int, 128>()));
	result &= REPORT_RESULT((test_feature<float, 128>()));
#endif

    return result;
}

