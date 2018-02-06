// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test double: assign double literal</summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

const int max_size = 6 * 5 * 4 * 3 * 2 * 1;

#define RESULT 3.1415926

void kernel(double & c) __GPU
{
    c = RESULT;
}

runall_result test_rank_dbl()
{
    int extent_data[] = {6, 5, 4, 3, 2, 1};
    const int _rank = 6;
    extent<_rank> e(extent_data);
    vector<double> data_in(max_size);
    vector<double> data_out(max_size);
    array<double, _rank> aA(e, data_in.begin(), data_in.end());

    parallel_for_each(aA.get_extent(), [&](index<_rank> idx) __GPU {
        kernel(aA[idx]);
    });

    data_out = aA;

    for (int i = 0; i < max_size; i++) {
        if (AreAlmostEqual(data_out[i], RESULT) == false) {
            Log(LogType::Error, true) << "Expected: " << RESULT << "get: " << data_out[i] << std::endl;
            return runall_fail;
        }
    }

    return runall_pass;
}

runall_result test()
{
    return test_rank_dbl();
}

runall_result test_main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double types.
    // Test is relying on default device, therefore check below is also done on default device.
    accelerator device = require_device_for<double>(device_flags::NOT_SPECIFIED, false);

    return test();
}

