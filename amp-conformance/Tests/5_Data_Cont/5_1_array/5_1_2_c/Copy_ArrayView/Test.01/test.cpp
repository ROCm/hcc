// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an array using copy constructor of form a(b) where b is array_view</summary>

#include "./../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
	{
		bool pass = test_array_copy_constructors_with_array_view<_type, _rank>();
		
		if (!pass)
            return false;
	}
	
	{
		bool pass = test_array_copy_constructors_with_array_view<_type, _rank,accelerator_view>((_gpu_device).get_default_view()) &&
					test_array_copy_constructors_with_array_view<_type, _rank,accelerator_view>((_gpu_device).create_view(queuing_mode_automatic)) &&
					test_array_copy_constructors_with_array_view<_type, _rank,accelerator_view>((_gpu_device).create_view(queuing_mode_immediate));
					
		if (!pass)
            return false;
	}
	
    {
        bool pass = test_array_copy_constructors_with_array_view<_type, _rank, accelerator_view>((_gpu_device).get_default_view(), _cpu_device.get_default_view()) &&
            test_array_copy_constructors_with_array_view<_type, _rank, accelerator_view>((_gpu_device).create_view(queuing_mode_immediate), _cpu_device.create_view(queuing_mode_immediate)) &&
            test_array_copy_constructors_with_array_view<_type, _rank, accelerator_view>((_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_immediate)) &&
            test_array_copy_constructors_with_array_view<_type, _rank, accelerator_view>((_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	SKIP_IF(!is_gpu_hardware_available());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 2>()));
	result &= REPORT_RESULT((test_feature<float, 3>()));
	result &= REPORT_RESULT((test_feature<double, 5>()));

    return result;
}

