// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator_view and staging buffer specialized construtors - between different gpu devices</summary>

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature(const std::vector<accelerator>& devices)
{
    for (size_t i = 0; i < devices.size()-1; i++)
    {
        accelerator pdevice1 = devices[i];
        accelerator pdevice2 = devices[i+1];
		WLog(LogType::Info, true) << "device1 = devices[" << i << "] = " << pdevice1.get_device_path() << std::endl;
		WLog(LogType::Info, true) << "device2 = devices[" << (i+1) << "] = " << pdevice2.get_device_path() << std::endl;

        {
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice1.get_default_view(), pdevice2.get_default_view()))
                return false;
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice1.create_view(queuing_mode_immediate), pdevice2.create_view(queuing_mode_automatic)))
                return false;
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice1.create_view(queuing_mode_automatic), pdevice2.create_view(queuing_mode_immediate)))
                return false;
            Log(LogType::Info, true) << "Finished - device1 for device2" << std::endl;
        }

        {
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice2.get_default_view(), pdevice1.get_default_view()))
                return false;
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice2.create_view(queuing_mode_immediate), pdevice1.create_view(queuing_mode_automatic)))
                return false;
            if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(pdevice2.create_view(queuing_mode_automatic), pdevice1.create_view(queuing_mode_immediate)))
                return false;
            Log(LogType::Info, true) << "Finished - device2 for device1" << std::endl;
        }
		Log(LogType::Info, true) << std::endl;
    }

	return true;
}

runall_result test_main()
{
	bool should_require_double = std::is_same<AMPTEST_T, double>::value;

    // get only gpu devices
	device_flags dflags = device_flags::NOT_EMULATED;
	if(should_require_double) {
		dflags |= device_flags::LIMITED_DOUBLE;
	}
    std::vector<accelerator> devices = get_available_devices(dflags);
    SKIP_IF(devices.size() < 2);

    return test_feature<AMPTEST_T, 5>(devices);
}

