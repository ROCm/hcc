// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests create_view() member</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;
    accelerator acc = Test::require_device(device_flags::NOT_SPECIFIED);

    accelerator_view av = acc.create_view();
    result &= REPORT_RESULT((av != acc.get_default_view()) == true);
    result &= REPORT_RESULT(av.get_queuing_mode() == queuing_mode_automatic);
    result &= REPORT_RESULT(is_accelerator_view_operable(av));

    av = acc.create_view(queuing_mode_immediate);
    result &= REPORT_RESULT(av.get_queuing_mode() == queuing_mode_immediate);
    result &= REPORT_RESULT(is_accelerator_view_operable(av));

    av = acc.create_view(queuing_mode_automatic);
    result &= REPORT_RESULT(av.get_queuing_mode() == queuing_mode_automatic);
    result &= REPORT_RESULT(is_accelerator_view_operable(av));

    return result;
}
