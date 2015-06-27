// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>AMP function with default parameter</summary>

#include "amptest.h"

using namespace concurrency;
using namespace concurrency::Test;

int test(int a, int b = 5)  __GPU
{
    return a + b;
}

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view av = device.get_default_view();

    int r = GPU_INVOKE(av, int, test, 3);
    return r == 8 ? runall_pass : runall_fail;
}
