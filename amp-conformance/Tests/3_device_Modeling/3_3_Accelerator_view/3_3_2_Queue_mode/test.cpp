// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests queue mode property of all accelerator views created by get_all API</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

runall_result test_main()
{
    runall_result result;

    std::vector<accelerator> accs = accelerator::get_all();
    std::for_each(accs.begin(), accs.end(),
        [&](accelerator& acc)
    {
        Log(LogType::Info, true) << "For device : " << acc.get_description() << std::endl;

        // default accelerator view
        Log(LogType::Info, true) << "Default view : " << std::endl;
        result &= REPORT_RESULT(acc.get_default_view().get_queuing_mode() == queuing_mode_automatic);
        result &= REPORT_RESULT(acc.get_default_view().get_queuing_mode() != queuing_mode_immediate);

        // immediate accelerator view
        Log(LogType::Info, true) << "Test view with immediate queue mode : " << std::endl;
        accelerator_view av_imm = acc.create_view(queuing_mode_immediate);
        result &= REPORT_RESULT(av_imm.get_queuing_mode() == queuing_mode_immediate);
        result &= REPORT_RESULT(av_imm.get_queuing_mode() != queuing_mode_automatic);

        // automatic accelerator view
        Log(LogType::Info, true) << "Test view with automatic queue mode : " << std::endl;
        accelerator_view av_auto = acc.create_view(queuing_mode_automatic);
        result &= REPORT_RESULT(av_auto.get_queuing_mode() == queuing_mode_automatic);
        result &= REPORT_RESULT(av_auto.get_queuing_mode() != queuing_mode_immediate);
    }
    );

    return result;
}
