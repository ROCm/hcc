// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Array(extent based) constructed with only begin iterator - use forward_list</summary>

#include <forward_list>
#include "./../../../constructor.h"
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::forward_list<_type> data;
    for (unsigned int i = 0; i < e1.size(); i++)
        data.push_front((_type)rand());

    {
        bool pass = test_feature_itr<_type, _rank>(e1, data.begin(), data.end()) &&
            test_feature_itr<_type, _rank>(e1, data.cbegin(), data.cend());

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	accelerator::set_default(require_device_for<AMPTEST_T>(device_flags::NOT_SPECIFIED, false).get_device_path());

	runall_result result;

	result &= REPORT_RESULT((test_feature<AMPTEST_T, 1>()));
	result &= REPORT_RESULT((test_feature<AMPTEST_T, 2>()));
	result &= REPORT_RESULT((test_feature<AMPTEST_T, 3>()));
	result &= REPORT_RESULT((test_feature<AMPTEST_T, 5>()));

	return result;
}

