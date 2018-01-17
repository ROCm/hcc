// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create array using 3D iterator specialized constructors - use set</summary>

#include <set>
#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _D0, int _D1, int _D2>
bool test_feature()
{
    {
        const int _rank = 3;
        std::set<_type> data;
        while (data.size() < _D0*_D1*_D2)
            data.insert((_type)rand());

        bool pass = test_feature_itr<_type, _rank, _D0, _D1, _D2>(data.begin(), data.end()) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2>(data.cbegin(), data.cend()) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2>(data.crbegin(), data.crend()) &&
                test_feature_itr<_type, _rank, _D0, _D1, _D2>(data.rbegin(), data.rend());

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double types.
	accelerator::set_default(require_device_with_double().get_device_path());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1, 1, 1>()));
	result &= REPORT_RESULT((test_feature<int, 7, 31, 2>()));
	result &= REPORT_RESULT((test_feature<unsigned int, 5, 91, 5>()));
    result &= REPORT_RESULT((test_feature<unsigned, 31, 19, 1>()));
	result &= REPORT_RESULT((test_feature<signed, 91, 5, 5>()));
    result &= REPORT_RESULT((test_feature<float, 2, 31, 19>()));
	result &= REPORT_RESULT((test_feature<float, 5, 1, 5>()));
    result &= REPORT_RESULT((test_feature<double, 13, 7, 7>()));
	
	return result;
}

