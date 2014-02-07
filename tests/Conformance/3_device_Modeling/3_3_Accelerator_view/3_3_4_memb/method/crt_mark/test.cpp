// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.

/// <summary>Tests create_marker member</summary>
// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include <amp.h>
#include "./../../../../../5_Data_Cont/amp.compare.h"
#include "./../../../../../5_Data_Cont/data.h"
#include "./../../../../../device.h"

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

void kernel_int_add(index<1> idx, array<int, 1> &fc, array<int, 1> &fa, array<int, 1> &fb) restrict(amp)
{
    fc[idx] = fa[idx] + fb[idx];
}

void init_data(size_t size, vector<int> A, vector<int> B, vector<int> C, vector<int> refC)
{
    Fill(A);
    Fill(B);

    for (size_t i = 0; i < size; i++)
    {
        refC[i] = A[i] + B[i];
    }
}

int main()
{
    int result = 1;
    accelerator_view av = require_device().get_default_view();

    const size_t size = 256;
    const size_t itr = 64;

    std::vector<int> A(size);
    std::vector<int> B(size);
    std::vector<int> C(size);
    std::vector<int> refC(size);

    init_data(size, A, B, C, refC);

    extent<1> e(size);
    array<int, 1> fA(e, A.begin(), av);
    array<int, 1> fB(e, A.begin(), av);
    array<int, 1> fC(e, av);

    for (size_t i = 0; i < itr; ++i)
    {
        parallel_for_each(e, [&](index<1> idx) restrict(amp) {
            kernel_int_add(idx, fC, fA, fB);
        });
    }

    auto ev = av.create_marker();

    ev.wait();

    C = fC;

    result = Verify(C, refC);

    return !result;
}
