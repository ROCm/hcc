// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests create_view() member</summary>
// RUN: %cxxamp %s %link
// RUN: ./a.out

#include "../../../accelerator.common.h"
#include "../../../../device.h"

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    int result = 1;
    accelerator acc = require_device();

    accelerator_view av = acc.create_view();
    result &= ((av != acc.get_default_view()) == true);
    result &= (av.get_queuing_mode() == queuing_mode_automatic);
    result &= (is_accelerator_view_operable(av));

    av = acc.create_view(queuing_mode_immediate);
    result &= (av.get_queuing_mode() == queuing_mode_immediate);
    result &= is_accelerator_view_operable(av);

    av = acc.create_view(queuing_mode_automatic);
    result &= (av.get_queuing_mode() == queuing_mode_automatic);
    result &= is_accelerator_view_operable(av);

    return !result;
}
