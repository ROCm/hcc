// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the accelerator set_default static member. This test assumes a default accelerator is available and is different from cpu accelerator. If assumption is invalid test will skip</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;

    // require_device() will cause test to skip
    // if no accelerators found other than cpu
    Test::require_device(device_flags::NOT_SPECIFIED);

    accelerator::set_default(accelerator::cpu_accelerator);
    accelerator acc;

    result &= REPORT_RESULT(acc == accelerator(accelerator::cpu_accelerator));

    return result;
}
