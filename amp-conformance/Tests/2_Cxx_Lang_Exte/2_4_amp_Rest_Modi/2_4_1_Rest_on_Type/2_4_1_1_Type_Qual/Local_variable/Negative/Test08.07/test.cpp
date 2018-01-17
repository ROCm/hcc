// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Neg: volatile pointers.</summary>
//#Expects: Error: test.cpp\(41\) : error C3581:.*(\bvolatile s &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(42\) : error C3581:.*(\bvolatile bool &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(43\) : error C3581:.*(\bvolatile int &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(44\) : error C3581:.*(\bvolatile double &).*:.*(unsupported type in amp restricted code)?

#include "common.h"

class s
{
public:
    int i;
    double d;
    unsigned long ul;
    float f;
};

bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        volatile s &ps = (volatile s &)aA[idx]; // not allowed here
        volatile bool &pb1 = (volatile bool&)aA[idx];
        volatile int &pi1 = (volatile int&)aA[idx];;
        volatile double &pd1 = (volatile double&)aA[idx];

        aA[idx] = 1;
    });

    for (int i =  0; i < size; i++)
    {
        if (A[i] != INIT_VALUE)
            return false;
    }

    return true;
}

runall_result test_main()
{
    bool passed = true;

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test(rv);

    printf("%s\n", "Failed!");

    return runall_fail;
}

