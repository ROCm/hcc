// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that operator precedence. Test nesting and chaining of operators.</summary>

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
    index<3> ia(5, 5, 5);
    index<3> ib(2, 2, 2);
    index<3> ic(3, 3, 3);
    index<3> ir, it;

    ir = index<3>(4, 4, 4);
    it = ia + ib - ic; // "*" > "+"

    if (!(it == ir))
    {
        return 11;   // test1 scenario1 failed
    }

    ir = index<3>(6, 6, 6);
    it = ia - ib + ic; // "*" > "-"

    if (!(it == ir))
    {
        return 12;   // test1 scenario2 failed
    }

    index<3> i1(1, 1, 1);
    index<3> i2(2, 2, 2);
    index<3> i3(3, 3, 3);
    index<3> i4(4, 4, 4);
    index<3> i5(5, 5, 5);
    index<3> i6(6, 6, 6);
    index<3> i7(7, 7, 7);
    index<3> i8(8, 8, 8);
    index<3> i9(9, 9, 9);
    
    ir = index<3>(-7, -7, -7);
    it = i1 + i2 + i3 - i4 - i5 + i6 + i7 - i8 - i9 ;

    if (!(it == ir))
    {
        return 13;   // test1 scenario3 failed
    }

    ir = index<3>(19, 19, 19);    
    it = (i1 + i2 ) + (i3 - i4 + i5) - i2 + (i6 + i7) - (i8 - i9);
    
    if (!(it == ir))
    {
        return 14;   // test1 scenario4 failed
    }

    return 0;   // all passed
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

    parallel_for_each(e, [&](index<1> idx) restrict(amp,cpu){
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < size; i++)
    {
        if (presult[i] != 0)
        {
            int ret = presult[i];
            return ret;
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

