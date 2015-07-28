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
/// <summary>Copy from an Array View (Array) to another Array View (C-array)</summary>

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    accelerator device = require_device(Device::ALL_DEVICES);
    accelerator_view acc_view = device.get_default_view();

    int size = 23;
    std::vector<float> src_v(size);
    Fill<float>(src_v);
    array<float> src_data(size, src_v.begin(), src_v.end(), acc_view);
    array_view<float, 1> src(src_data);

    std::vector<float> dest_v(size);
    array_view<float> dest(size, dest_v);

    src.copy_to(dest);
    return VerifyDataOnCpu(dest, src) ? runall_pass : runall_fail;
}

