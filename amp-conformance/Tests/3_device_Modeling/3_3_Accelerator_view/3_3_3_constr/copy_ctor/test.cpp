// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests that using copy constructor on a CPU accelerator view creates a valid and operable copy</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;

runall_result test_main()
{
    runall_result result;

    accelerator acc(accelerator::cpu_accelerator);
    accelerator_view av_copy(static_cast<const accelerator_view&>(acc.get_default_view()));

    result &= REPORT_RESULT(is_accelerator_view_equal(acc.get_default_view(), av_copy));
    result &= REPORT_RESULT(is_accelerator_view_operable(av_copy));

    return result;
}
