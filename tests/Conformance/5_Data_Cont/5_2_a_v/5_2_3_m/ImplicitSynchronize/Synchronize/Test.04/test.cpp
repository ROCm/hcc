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
/// <summary>Verify a manual asynchronous synchronization</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include"../../../../../../device.h"
using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    accelerator acc = require_device();
	
    if(acc.get_supports_cpu_shared_memory())
    {
        acc.set_default_cpu_access_type(access_type_read_write);
    }
	
    accelerator::set_default(acc.get_device_path());
    
    std::vector<int> v(10);
    std::fill(v.begin(), v.end(), 5);
	
    array_view<int, 1> av(10, v);
	
    parallel_for_each(extent<1>(1), [=](index<1>) restrict(amp) {
        av[0] = 17;
    });

    std::shared_future<void> w = av.synchronize_async();
    w.wait();

    return !(v[0] == 17);
}
