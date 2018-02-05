// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator and staging buffer specialized construtors - 2 arrays with different view of same device </summary>

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
runall_result test_feature()
{
	vector<accelerator> devices = get_available_devices(device_flags::NOT_SPECIFIED); // all but CPU devices

	runall_result result;
	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator& device = devices[i];

		WLog(LogType::Info, true) << "Device " << i << ": " << device.get_description() << " (" << device.get_device_path() << ")" << std::endl;
		Log(LogType::Info, true) << "  Version: " << device.get_version() << "; Memory: " << device.get_dedicated_memory() << std::endl;
		Log(LogType::Info, true) << "  Debug: " << device.get_is_debug() << "; Emulated: " << device.get_is_emulated() << "; Has Display: " << device.get_has_display() << std::endl;

        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(device.get_default_view(), device.get_default_view())));
        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(device.create_view(queuing_mode_immediate), device.create_view(queuing_mode_automatic))));
        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(device.create_view(queuing_mode_automatic), device.create_view(queuing_mode_immediate))));
		Log_writeline(LogType::Info);
	}

	return result;
}

template<int _rank>
runall_result test_feature()
{
	vector<accelerator> devices = get_available_devices(device_flags::LIMITED_DOUBLE);
	if(devices.size() == 0) {
		return runall_skip;
	}

	runall_result result;
	for (size_t i = 0; i < devices.size(); i++)
	{
		accelerator& device = devices[i];

		WLog(LogType::Info, true) << "Device " << i << ": " << device.get_description() << " (" << device.get_device_path() << ")" << std::endl;
		Log(LogType::Info, true) << "  Version: " << device.get_version() << "; Memory: " << device.get_dedicated_memory() << std::endl;
		Log(LogType::Info, true) << "  Debug: " << device.get_is_debug() << "; Emulated: " << device.get_is_emulated() << "; Has Display: " << device.get_has_display() << std::endl;

        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<double, _rank, accelerator_view>(device.get_default_view(), device.get_default_view())));
        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<double, _rank, accelerator_view>(device.create_view(queuing_mode_immediate), device.create_view(queuing_mode_automatic))));
        result &= REPORT_RESULT((test_accl_staging_buffer_constructor<double, _rank, accelerator_view>(device.create_view(queuing_mode_automatic), device.create_view(queuing_mode_immediate))));
		Log_writeline();
	}

	return result;
}

runall_result test_main()
{
	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 3>()));
	result &= REPORT_RESULT((test_feature<float, 4>()));

	// test with double
	runall_result dbl_result = REPORT_RESULT(test_feature<8>());
	if(!dbl_result.get_is_skip()) // don't aggregate if skipped
		result &= dbl_result;

    return result;
}

