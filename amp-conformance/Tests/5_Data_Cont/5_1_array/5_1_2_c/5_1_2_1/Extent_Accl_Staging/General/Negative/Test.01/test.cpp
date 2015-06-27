// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Exception when using gpu as staging buffer for CPU using default_view - this is due to amp-restricted parallel_for_each</summary>

#include "./../../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    test_accl_staging_buffer_constructor<_type, _rank, accelerator_view>(_cpu_device.get_default_view(), (_gpu_device).get_default_view());

	return false;
}

runall_result test_main()
{
	SKIP_IF(!is_gpu_hardware_available());

    try
    {
        test_feature<int, 5>();
    }
	catch (runtime_exception &ex)
	{
		return runall_pass;
	}
    catch (std::exception e)
    {
        return runall_fail;
    }

    return runall_fail;
}

