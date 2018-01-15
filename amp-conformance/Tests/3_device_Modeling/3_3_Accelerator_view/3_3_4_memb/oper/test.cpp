// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests the assignment operator works the same as copy constructor and tests logical operators. This test assumes a default accelerator is available and is different from cpu accelerator.If assumption is invalid test will skip</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;

    accelerator acc = Test::require_device(device_flags::NOT_SPECIFIED);
    accelerator_view av = acc.get_default_view();
    accelerator_view av_assign = acc.create_view();
    accelerator_view av_copy(av);
    av_assign = av;

    // verify = operator works same as copy constr
    result &= REPORT_RESULT(av_assign == av_copy);
    result &= REPORT_RESULT(is_accelerator_view_equal(av_assign, av_copy));
    result &= REPORT_RESULT(run_simple_p_f_e(av_assign));

    accelerator acc_cpu = accelerator(accelerator::cpu_accelerator);
    accelerator_view av_cpu = acc_cpu.get_default_view();
    accelerator_view av_assign_cpu = acc_cpu.create_view();
    av_assign_cpu = av_cpu;

    // verify = operator generate operable and equal view for cpu accelecrator
    result &= REPORT_RESULT(av_assign_cpu == acc_cpu.get_default_view());
    result &= REPORT_RESULT(is_accelerator_view_equal(acc_cpu.get_default_view(), av_assign_cpu));
    result &= REPORT_RESULT(is_accelerator_view_operable(av_assign_cpu));

    // verify == operator
    result &= REPORT_RESULT((av_assign_cpu == acc_cpu.get_default_view()) == true);
    result &= REPORT_RESULT((av_assign_cpu == av_copy) == false);

    // verify != operator
    result &= REPORT_RESULT((av_assign_cpu != av_copy) == true);
    result &= REPORT_RESULT((av_assign_cpu != acc_cpu.get_default_view()) == false);

    return result;
}
