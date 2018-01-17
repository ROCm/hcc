// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator_view specialized constructors - using default view.</summary>

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
int test_feature()
{
    vector<accelerator> devices = accelerator::get_all();

    printf("Found %lu devices\n", devices.size());

    for (size_t i = 0; i < devices.size(); i++)
    {
        if (devices[i] == _cpu_device)
            continue;

        accelerator device = devices[i];

		printf("Device %zu: %ls (%ls)\n", i, device.get_description().c_str(), device.get_device_path().c_str());
        printf("Version %d \t Memory %zu\n", device.get_version(), device.get_dedicated_memory());
        printf("Debug:%c \t Emulated:%c \t Display: %c\n", device.get_is_debug() ? 'Y' : 'N', device.get_is_emulated() ? 'Y' : 'N', device.get_has_display() ? 'Y' : 'N');

        test_accl_constructor<_type, _rank, accelerator_view>(device.get_default_view());

        printf("Finished with device %zu\n", i);
    }

    return runall_pass;
}


template<int _rank>
int test_feature()
{
    vector<accelerator> devices = accelerator::get_all();

    printf("Found %lu devices\n", devices.size());

    for (size_t i = 0; i < devices.size(); i++)
    {
        if (devices[i] == _cpu_device)
            continue;

        accelerator device = devices[i];

        if (!device.get_supports_double_precision())
        {
            return runall_skip;
        }

		printf("Device %zu: %ls (%ls)\n", i, device.get_description().c_str(), device.get_device_path().c_str());
        printf("Version %d \t Memory %zu\n", device.get_version(), device.get_dedicated_memory());
        printf("Debug:%c \t Emulated:%c \t Display: %c\n", device.get_is_debug() ? 'Y' : 'N', device.get_is_emulated() ? 'Y' : 'N', device.get_has_display() ? 'Y' : 'N');

        test_accl_constructor<double, _rank, accelerator_view>(device.get_default_view());

        printf("Finished with device %zu\n", i);
    }

    return runall_pass;
}

runall_result test_main()
{
	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 5>()));
	result &= REPORT_RESULT((test_feature<float, 5>()));

	// test with double
	runall_result dbl_result = REPORT_RESULT(test_feature<5>());
	if(!dbl_result.get_is_skip()) // don't aggregate if skipped
		result &= dbl_result;

    return result;
}

