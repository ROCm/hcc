// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that accessing each dimension returns the correct index component</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;

int test() restrict(amp,cpu)
{
    index<1> i1(100);

    if (i1[0] != 100)
    {
        return 11;
    }

    index<3> i3(100, 200, 300);

    if ((i3[0] != 100) ||(i3[1] != 200) ||(i3[2] != 300))
    {
        return 12;
    }

    int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    index<10> i10(data);

    for (int i = 0; i < 10; i++)
    {
        if (i10[i] != i + 1)
        {
            return 13;
        }
    }

    return 0;
}

int main()
{
    int result = 1;

    result &= (test() == 0);
    return !result;
}
