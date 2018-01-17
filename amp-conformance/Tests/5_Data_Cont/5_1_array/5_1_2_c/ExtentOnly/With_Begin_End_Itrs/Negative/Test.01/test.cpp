// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify runtime behaviour when data container/iterator has more data than specified in extent - use set</summary>

#include<set>

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    {
        int edata[_rank];
        for (int i = 0; i < _rank; i++)
            edata[i] = i+1;
        extent<_rank> e1(edata);

        std::set<_type> data;
        for (unsigned int i = 0; i < e1.size()+1; i++)
            data.insert((_type)rand());

        bool pass = test_feature_itr<_type, _rank>(e1, data.begin(), data.end());

        if (!pass)
            return false;
    }

    return true;
}

runall_result test_main()
{
    try
    {
        test_feature<int, 5>();
    }
	catch(runtime_exception &ex)
	{
		return runall_pass;
	}
    catch (std::exception e)
    {
        return runall_fail;
    }

    return runall_fail;
}

