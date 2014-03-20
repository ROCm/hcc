// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests if the default accelerator constructor creates the default accelerator</summary>

#include <amptest.h>
#include <amptest_main.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;

runall_result test_main()
{
    runall_result result;

    accelerator acc;
    accelerator acc_expected(accelerator::default_accelerator);

    const wchar_t literal_string[256] = L"default";

    accelerator acc_literal_string(static_cast<const wchar_t*>(literal_string));

    result &= REPORT_RESULT(is_accelerator_equal(acc, acc_expected));
    result &= REPORT_RESULT(is_accelerator_view_operable(acc.get_default_view()));
    result &= REPORT_RESULT(is_accelerator_equal(acc, acc_literal_string));
    result &= REPORT_RESULT(is_accelerator_view_operable(acc_literal_string.get_default_view()));

    return result;
}
