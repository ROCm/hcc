// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check for addition and muplication operators, a @ b is equal to b @ a.</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;

int test() restrict(amp,cpu)
{
    int data1[] = {-100, -10, -1, 0,  1,  10, 100};
    int data2[] = {  99,  10,  1, 1, -1, -10, -9};
    index<7> i1(data1);
    index<7> i2(data2);
    index<7> ir1;
    index<7> ir2;

    ir1 = i1 + i2;
    ir2 = i2 + i1;

    if (!(ir1 == ir2))
    {
        return 11;  //test1 scenario1 failed
    }

    return 0;
}

int main()
{
    int result = 1;

    result &= test();
    return result;
}
