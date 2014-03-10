// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that when == returns true then != returns false.</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;
using std::vector;

int test() restrict(amp,cpu)
{
    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-10, -1, 0, 1, 10};

    index<5> i1(data1);
    index<5> i2(data2);

    if ((i1 == i2) && (!(i1 != i2)))
    {
        return 0;
    }

    return 1;
}

void kernel(index<1>& idx, array<int, 1>& result) restrict(amp,cpu)
{
    result[idx] = test();    
}

const int size = 10;

int test_device()
{
    extent<1> e(size);
    array<int, 1> result(e);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) restrict(amp,cpu) {
        kernel(idx, result);
    });

    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] == 1)
        {
            return 1;
        }
    }

    return 0;
}

int main() 
{ 
	 int result = 1;
    // Test on host
    result &= (test() == 0);
   
    // Test on device
	 result &= (test_device() == 0);
    return !result;
}
