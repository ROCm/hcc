// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>M3</tags>
/// <summary>Create an extern "C" function and verify it can be called from CPU or GPU (as appropriate)</summary>

#include "amptest.h"

using namespace concurrency;
using namespace concurrency::Test;

extern "C" int gpu_only() __GPU_ONLY
{
    return 2;
}

extern "C" int cpu_only() __CPU_ONLY_EXPLICIT
{
    return 1;
}

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view av = device.get_default_view();

    int r;
    Log(LogType::Info, true) << "Executing gpu_only on the GPU" << std::endl;
    r = GPU_INVOKE(av, int, gpu_only);
    if (r != 2)
    {
        Log(LogType::Info, true) << "Value was: " << r << "Expected 2" << std::endl;
        return runall_fail;
    }

    Log(LogType::Info, true) << "Executing cpu_only on the CPU" << std::endl;
    r = cpu_only();
    if (r != 1)
    {
        Log(LogType::Info, true) << "Value was: " << r << "Expected 1" << std::endl;
        return runall_fail;
    }

    Log(LogType::Info, true) << "Passed" << std::endl;
    return runall_pass;
}

