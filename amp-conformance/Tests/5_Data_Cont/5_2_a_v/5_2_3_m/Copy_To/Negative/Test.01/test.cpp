//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Copy an array_view to an array with incompatible extent</summary>

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    extent<2> av_ex(20, 10);
    vector<long> vec(av_ex.size());
    Fill<long>(vec.data(), av_ex.size());
    array_view<long, 2> av(av_ex, vec);

    // the array has a different shape
    extent<2> arr_ex(10, 20);
    array<long, 2> arr(arr_ex, acc_view);

    try
    {
        // this should throw
        av.copy_to(arr);
        return runall_fail;
    }
    catch (runtime_exception &e)
    {
        Log(LogType::Info, true) << e.what() << std::endl;
        return runall_pass;
    }
}

