// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Array(extent based) constructed with only begin iterator - use unordered_multiset</summary>

#include <unordered_set>
#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::unordered_multiset<_type> data;
    for (unsigned int i = 0; i < e1.size(); i++)
        data.insert((_type)rand());

    {
        bool pass = test_feature_itr<_type, _rank>(e1, data.begin());

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	accelerator::set_default(require_device(device_flags::NOT_SPECIFIED).get_device_path());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1>()));
	result &= REPORT_RESULT((test_feature<int, 2>()));
	result &= REPORT_RESULT((test_feature<int, 3>()));
	result &= REPORT_RESULT((test_feature<int, 5>()));

	result &= REPORT_RESULT((test_feature<unsigned, 1>()));
	result &= REPORT_RESULT((test_feature<unsigned, 2>()));
	result &= REPORT_RESULT((test_feature<unsigned, 3>()));
	result &= REPORT_RESULT((test_feature<unsigned, 5>()));

	result &= REPORT_RESULT((test_feature<float, 1>()));
	result &= REPORT_RESULT((test_feature<float, 2>()));
	result &= REPORT_RESULT((test_feature<float, 3>()));
	result &= REPORT_RESULT((test_feature<float, 5>()));

	result &= REPORT_RESULT((test_feature<double, 1>()));
	result &= REPORT_RESULT((test_feature<double, 2>()));
	result &= REPORT_RESULT((test_feature<double, 3>()));
	result &= REPORT_RESULT((test_feature<double, 5>()));

	return result;
}

