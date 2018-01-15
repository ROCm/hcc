// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Array(extent based) constructed with bounded iterators - use unordered_multiset</summary>

#include <unordered_set>
#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _D0, int _D1>
bool test_feature()
{
    const int _rank = 2;

    {
        std::unordered_multiset<_type> data;
        while (data.size() < (_D0 * _D1))
            data.insert((_type)rand());

        bool pass = test_feature_itr<_type, _rank, _D0, _D1>(data.begin(), data.end()) &&
            test_feature_itr<_type, _rank, _D0, _D1>(data.cbegin(), data.cend());

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
	accelerator::set_default(require_device(device_flags::NOT_SPECIFIED).get_device_path());

	runall_result result;

	result &= REPORT_RESULT((test_feature<int, 1, 1>()));

	return result;
}

