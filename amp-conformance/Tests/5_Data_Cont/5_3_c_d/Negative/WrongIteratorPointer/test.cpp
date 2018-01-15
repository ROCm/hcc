// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Copying between standard container and destination with different types</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    accelerator cpuDevice(accelerator::cpu_accelerator);

    std::vector<int> cont(3);
    std::fill(cont.begin(), cont.end(), 5);
    array<int, 1> destArray(3, cpuDevice.get_default_view());

    try
    {
        copy(cont.end(), cont.begin(), destArray);
    }
    catch(runtime_exception)
    {
        return runall_pass;
    }

    return runall_fail;
}

