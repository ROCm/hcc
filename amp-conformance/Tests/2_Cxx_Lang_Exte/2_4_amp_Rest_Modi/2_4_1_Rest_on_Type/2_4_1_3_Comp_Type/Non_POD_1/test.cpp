// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test that non-POD array is supported in amp restriction</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

class NonPodClass
{
private:
    int var;

public:
    NonPodClass()__GPU_ONLY
    {
        var = 10;
    }

    NonPodClass(int i)__GPU_ONLY
    {
        var = i;
    }

    int get_var() const __GPU_ONLY
    {
        return var;
    }
};

runall_result Test1() __GPU_ONLY
{
    NonPodClass arr1[5];

    return (arr1[0].get_var() == 10) ? runall_pass : runall_fail;
}

int main()
{
    accelerator_view av = require_device(Device::ALL_DEVICES).get_default_view();

    runall_result result = GPU_INVOKE(av, runall_result, Test1);

    Log(LogType::Info, true) << result.get_name() << std::endl;
    return result.get_exit_code();
}

