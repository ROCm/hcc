//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>- Create an array_view of type const using extent value, e0, e1 and e2, and a container in a CPU restricted function. </summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>
#include "../../helper.h"

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

runall_result test_main()
{
    const int m = 100, n = 80, o = 10;
    const int size = m * n * o;

    vector<int> vec1(size);
    Fill<int>(vec1.data(), size);
    vector<int> vec2(vec1);

    array_view<const int, 3> av1(m, n, o, vec1);
    array_view<int, 3> av2(m, n, o, vec2);

    if(m != av1.get_extent()[0]) // Verify extent
    {
        Log(LogType::Error, true) << "array_view extent[0] different from extent used to initialize object." << std::endl;
        Log(LogType::Error, true) << "Expected: [" << m << "] Actual : [" << av1.get_extent()[0] << "]" << std::endl;
        return runall_fail;
    }

    if(n != av1.get_extent()[1]) // Verify extent
    {
        Log(LogType::Error, true) << "array_view extent[1] different from extent used to initialize object." << std::endl;
        Log(LogType::Error, true) << "Expected: [" << n << "] Actual : [" << av1.get_extent()[1] << "]" << std::endl;
        return runall_fail;
    }

    if(o != av1.get_extent()[2]) // Verify extent
    {
        Log(LogType::Error, true) << "array_view extent[2] different from extent used to initialize object." << std::endl;
        Log(LogType::Error, true) << "Expected: [" << o << "] Actual : [" << av1.get_extent()[2] << "]" << std::endl;
        return runall_fail;
    }

    // verify data
    if(!compare(vec1, av1))
    {
         Log(LogType::Error, true) << "array_view and vector data do not match" << std::endl;
         return runall_fail;
    }

    accelerator_view acc_view = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av1.get_extent(), [=](index<3> idx) restrict(amp)
    {
        av2[idx] = av1[idx] + 1;
    });

    // vec should be updated after this
    Log(LogType::Info, true) << "Accessing first element of array_view [value = " << av2(0,0,0) << "] to force synchronize." << std::endl;

    // verify data
	bool passed = true;
    for(int i = 0; i < size; i++)
    {
        if(vec2[i] != vec1[i] + 1)
        {
			Log(LogType::Error, true) << compose_incorrect_element_message(i, vec1[i] + 1, vec2[i]) << std::endl;
            passed = false;
        }
    }

    return passed;
}

