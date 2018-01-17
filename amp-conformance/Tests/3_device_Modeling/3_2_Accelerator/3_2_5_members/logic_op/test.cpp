// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the logical operator on an assigned accelerator and a copied accelerator. This test assumes a default accelerator is available and is different from cpu accelerator.If assumption is invalid test will skip</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;

    accelerator acc_def = Test::require_device(device_flags::NOT_SPECIFIED);
    accelerator acc = accelerator(accelerator::cpu_accelerator);

    // verify the logical operators using a copy constructed accelerator
    result &= REPORT_RESULT((acc == accelerator(accelerator::cpu_accelerator)) == true);
    result &= REPORT_RESULT((acc != accelerator(accelerator::cpu_accelerator)) == false);

    acc = acc_def;

    // verify the logical operators using an assign constructed accelerator
    result &= REPORT_RESULT((acc == acc_def) == true);
    result &= REPORT_RESULT((acc != acc_def) == false);

    return result;
}
