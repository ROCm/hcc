// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that when == returns true then != returns false.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-10, -1, 0, 1, 10};

    index<5> i1(data1);
    index<5> i2(data2);

    if ((i1 == i2) && (!(i1 != i2)))
    {
        return 0;
    }

    return 1;
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    extent<1> e(size);
    array<int, 1> result(e);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) __GPU {
        kernel(idx, result);
    });

    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] == 1)
        {
            return 1;
        }
    }

    return 0;
}

runall_result test_main()
{
	runall_result result;
    // Test on host
	Log(LogType::Info, true) << "Test == and != on host" << std::endl;
    result &= REPORT_RESULT(test() == 0);

    // Test on device
	Log(LogType::Info, true) << "Test == and != on device" << std::endl;
	result &= REPORT_RESULT(test_device() == 0);
    return result;
}

