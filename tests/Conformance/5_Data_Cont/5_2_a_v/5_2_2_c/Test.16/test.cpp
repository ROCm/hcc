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
/// <summary>Test that a const array_view can be copy constructed from a read-write array_view</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include <amp.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using std::vector;

int main()
{
    const int size = 100;

    vector<int> vec_rw(size), vec_ro(size);
    for(int i = 0; i < size; i++)
    {
        vec_rw[i] = i;
        vec_ro[i] = i;
    }

    array_view<int, 1> av_base(size, vec_ro); // rw array_view
    array_view<const int, 1> av_ro(av_base); // copy construct  ro array_view from rw array_view

    array_view<int, 1> av_rw(size, vec_rw); // for verification

    if(av_ro.get_extent()[0] != av_base.get_extent()[0]) // verify extent
    {
        return 1;
    }

    // use in parallel_for_each
    parallel_for_each(av_ro.get_extent(), [=] (index<1> idx) restrict(amp, cpu)
    {
        av_rw[idx] = idx[0] + 1;
    });
    
    // verify data
    for(int i = 0; i < size; i++)
    {
        if(av_rw[i] != (i + 1) || vec_rw[i] != (i + 1))
        {
            return 1;
        }
    }

    return 0;
}

