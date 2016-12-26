// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Construct array using accelerator_view and staging buffer specialized construtors - between  CPU and device</summary>

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
runall_result test_feature(accelerator *p_device_gpu,accelerator *p_device_cpu)
{
    Log(LogType::Info, true) << "Testing _type:" << typeid(_type).name() << " _rank:" << _rank << std::endl;

    if (p_device_cpu && p_device_gpu)
    {
        if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>((*p_device_gpu).create_view(queuing_mode_automatic), (*p_device_cpu).create_view(queuing_mode_automatic)))
            return runall_fail;
        if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>((*p_device_gpu).create_view(queuing_mode_immediate), (*p_device_cpu).create_view(queuing_mode_automatic)))
            return runall_fail;
        if (!test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>((*p_device_gpu).create_view(queuing_mode_automatic), (*p_device_cpu).create_view(queuing_mode_immediate)))
            return runall_fail;
    }

	Log(LogType::Info, true) << "Done" << std::endl;
	return runall_pass;
}

runall_result test_main()
{
    runall_result result;

    accelerator *p_device_cpu = NULL;
    accelerator *p_device_gpu = NULL;

    std::vector<accelerator> accl_devices = accelerator::get_all();

    for (size_t i = 0; i < accl_devices.size(); i ++)
    {
        if ((accl_devices[i].get_device_path() == accelerator::cpu_accelerator) && (!p_device_cpu))
        {
            p_device_cpu = &accl_devices[i];
        }
        else
        {
            p_device_gpu = &accl_devices[i];
        }
    }
    result &= REPORT_RESULT((test_feature<int, 5>(p_device_gpu,p_device_cpu)));
    result &= REPORT_RESULT((test_feature<unsigned int, 5>(p_device_gpu,p_device_cpu)));
    result &= REPORT_RESULT((test_feature<float, 5>(p_device_gpu,p_device_cpu)));

    return result;
}

