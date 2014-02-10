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
/// <summary>Copy from an Array View<const T> to an Array</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %llc -march=c -o %t/kernel_.cl < %t.ll
// RUN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include <vector>
#include"../../../../data.h"
#include"../../../../amp.compare.h"
using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    //accelerator device = require_device(Device::ALL_DEVICES);
    //accelerator_view acc_view = device.default_view;
    accelerator def;
    accelerator_view acc_view = def.get_default_view();


    int size = 23;
    std::vector<float> src_v(size);
    Fill<float>(src_v);
    array<float, 1> src_arr(size, src_v.begin(), src_v.end(), acc_view);
    array_view<float, 1> src(src_arr);

    array<float> dest(size, acc_view);
    src.copy_to(dest);

    return VerifyDataOnCpu(dest, src) ? 0 : 1;
}

