// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using array's extent based constructor - extents is 0</summary>

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    {
        std::unique_ptr<int[]> edata(new int[_rank]);
        for (int i = 0; i < _rank; i++)
            edata[i] = (i == 0) ? 0 : 1;

        extent<_rank> e1(edata.get());
        array<_type, _rank> src(e1);

        if (src.get_extent() != e1)
        {
            return false;
        }

        // verify array extents are modified
        for (int i = 0; i < _rank; i++)
        {
            if (edata[i] != src.get_extent()[i])
                return false;
        }
    }

    return true;
}

runall_result test_main()
{
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

