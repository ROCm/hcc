// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create array using 3D based specialized constructors - use set</summary>

#include <set>
#include <iostream>
#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _D0, int _D1, int _D2>
bool test_feature()
{
    const int _rank = 3;

    std::set<_type> data;
    for (int i = 0; i < _D0*_D1*_D2; i++)
        data.insert((_type)rand());

    {
        bool pass = test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(),data.end(), (_gpu_device).get_default_view()) &&
            test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(),data.end(), (_gpu_device).create_view(queuing_mode_immediate)) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(),data.end(), (_gpu_device).create_view(queuing_mode_automatic)) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(),data.end(), (_gpu_device).create_view(queuing_mode_automatic));

        if (!pass)
            return false;
    }

    {
        // After bug fix use the commented code as last parameter
        bool pass = test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(), (_gpu_device).get_default_view()) &&
            test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(), (_gpu_device).create_view(queuing_mode_immediate)) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(), (_gpu_device).create_view(queuing_mode_automatic)) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2, accelerator_view>(data.begin(), (_gpu_device).create_view(queuing_mode_immediate));

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	SKIP_IF(!is_gpu_hardware_available());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1, 1, 1>()));
	result &= REPORT_RESULT((test_feature<int, 7, 3, 2>()));
	result &= REPORT_RESULT((test_feature<int, 5, 3, 5>()));
	result &= REPORT_RESULT((test_feature<unsigned, 3, 2, 1>()));
	result &= REPORT_RESULT((test_feature<signed, 3, 5, 5>()));
	result &= REPORT_RESULT((test_feature<float, 2, 3, 2>()));
    result &= REPORT_RESULT((test_feature<float, 5, 1, 5>()));
	result &= REPORT_RESULT((test_feature<double, 2, 7, 7>()));
	
    return result;
}

