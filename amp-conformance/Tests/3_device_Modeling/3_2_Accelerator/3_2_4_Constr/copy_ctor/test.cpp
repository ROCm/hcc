// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the accelerator copy constructor</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;

runall_result test_main()
{
    runall_result result;

    accelerator acc(accelerator::cpu_accelerator);
    accelerator acc_copy(acc);

    result &= REPORT_RESULT(is_accelerator_equal(acc, acc_copy));
    result &= REPORT_RESULT(is_accelerator_view_operable(acc_copy.get_default_view()));

    return result;
}
