// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create array using 2-D accelerator_view and staging specialized constructors - uses unordered_multiset - GPU Host CPU target</summary>

#include <unordered_set>
#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _D0, int _D1>
bool test_feature()
{
    const int _rank = 2;

    std::unordered_multiset<_type> data;
    for (int i = 0; i < _D0*_D1; i++)
        data.insert((_type)rand());

    {
        bool pass = test_feature_staging_itr<_type, _rank, _D0, _D1, accelerator_view>(data.begin(), data.end(), (_gpu_device).get_default_view(), _cpu_device.get_default_view()) &&
            test_feature_staging_itr<_type, _rank, _D0, _D1, accelerator_view>(data.cbegin(), data.cend(), (_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }

    {
        bool pass = test_feature_staging_itr<_type, _rank, _D0, _D1, accelerator_view>(data.begin(), (_gpu_device).get_default_view(), _cpu_device.get_default_view()) &&
            test_feature_staging_itr<_type, _rank, _D0, _D1, accelerator_view>(data.cbegin(), (_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	SKIP_IF(!is_gpu_hardware_available());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1, 1>()));
	result &= REPORT_RESULT((test_feature<int, 7, 31>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 91, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned, 31, 19>()));
	result &= REPORT_RESULT((test_feature<signed, 91, 5>()));
	result &= REPORT_RESULT((test_feature<float, 31, 19>()));
    result &= REPORT_RESULT((test_feature<float, 5, 1>()));
	result &= REPORT_RESULT((test_feature<double, 13, 7>()));
	
    return result;
}

