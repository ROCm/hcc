// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create array using extent accelerator_view staging specialized constructors - uses multiset - CPU host GPU target</summary>

#include <set>
#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::multiset<_type> data;
    for (unsigned int i = 0; i < e1.size(); i++)
    {
        _type var = (_type)rand();
        data.insert(var);
    }

    {
        bool pass = test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.begin(), data.end(), (_gpu_device).get_default_view(), _cpu_device.get_default_view()) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.rbegin(), data.rend(), (_gpu_device).create_view(queuing_mode_immediate), _cpu_device.create_view(queuing_mode_immediate)) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.cbegin(), data.cend(), (_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_immediate)) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.crbegin(), data.crend(), (_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }

    {
        bool pass = test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.begin(), (_gpu_device).get_default_view(), _cpu_device.get_default_view()) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.rbegin(), (_gpu_device).create_view(queuing_mode_immediate), _cpu_device.create_view(queuing_mode_immediate)) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.cbegin(), (_gpu_device).create_view(queuing_mode_immediate), _cpu_device.create_view(queuing_mode_automatic)) &&
            test_feature_staging_itr<_type, _rank, accelerator_view>(e1, data.crbegin(), (_gpu_device).create_view(queuing_mode_automatic), _cpu_device.create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }


    return true;
}

runall_result test_main()
{
	SKIP_IF(!is_gpu_hardware_available());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 5>()));
	result &= REPORT_RESULT((test_feature<float, 5>()));
	result &= REPORT_RESULT((test_feature<double, 5>()));

    return result;
}

