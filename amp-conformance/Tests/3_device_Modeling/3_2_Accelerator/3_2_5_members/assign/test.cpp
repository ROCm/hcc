// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the assignment operator chaining and that it produce same result as copy constructor. This test assumes a default accelerator is available and is different from cpu accelerator.If assumption is invalid test will skip</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;

    accelerator acc_ref = Test::require_device(device_flags::NOT_SPECIFIED);
    accelerator acc1 = accelerator(accelerator::cpu_accelerator);
    accelerator acc2 = accelerator(accelerator::cpu_accelerator);
    accelerator acc3 = accelerator(accelerator::cpu_accelerator);

    acc1 = acc2 = acc3 = acc_ref;

    // verify that = operator produce a copy accelerator that is operable
    result &= REPORT_RESULT(acc1 == acc_ref);
    result &= REPORT_RESULT(acc2 == acc_ref);
    result &= REPORT_RESULT(acc3 == acc_ref);
    result &= REPORT_RESULT(run_simple_p_f_e(acc1.get_default_view()));
    result &= REPORT_RESULT(run_simple_p_f_e(acc2.get_default_view()));
    result &= REPORT_RESULT(run_simple_p_f_e(acc3.get_default_view()));

    // verify that = operator have same result as copy constructor
    accelerator acc_copy(acc_ref);
    result &= REPORT_RESULT(acc_copy == acc1);

    return result;
}
