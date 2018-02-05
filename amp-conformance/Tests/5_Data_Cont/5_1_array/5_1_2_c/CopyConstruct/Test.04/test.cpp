// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verifies that the copy constructor will copy pending writes to data</summary>

#include <amptest.h>
#include <vector>
#include <ostream>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

runall_result test_main()
{
    runall_result result;

    accelerator_view acc_view = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    int size = 20;
    std::vector<float> src_v(size);
    Fill(src_v);

    array<float, 1> src_arr(size, src_v.begin(), acc_view);

    // create a new array and copy data
    array<float> dest1(size, acc_view);
    src_arr.copy_to(dest1);

    // now copy construct from the new copy of the data
    array<float> dest2(dest1);

    array_view<float> av(dest1);
    Log(LogType::Info, true) << "Verifying original array" << std::endl;
    result &= REPORT_RESULT(VerifyDataOnCpu(av, src_v));

    av = array_view<float>(dest2);
    Log(LogType::Info, true) << "Now verifying copy constructed array" << std::endl;
    result &= REPORT_RESULT(VerifyDataOnCpu(av, src_v));

	return result;
}

