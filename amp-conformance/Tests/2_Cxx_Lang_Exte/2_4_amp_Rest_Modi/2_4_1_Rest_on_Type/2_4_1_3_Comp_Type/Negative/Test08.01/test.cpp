// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) (Use reference.) Define pointers to non–amp–compatible type in kernel function </summary>
//#Expects: Error: test.cpp\(32\) : error C3581:.*(\bchar &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(33\) : error C3581:.*(\bshort &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(34\) : error C3581:.*(\blong double &).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(35\) : error C3581:.*(\bwchar_t &).*:.*(unsupported type in amp restricted code)?

#include "common.h"

bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = 1;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        char &p1 = (char &)aA[idx]; // not allowed here
        short int &p2 = (short int &)aA[idx];
        long double &p4 = (long double &)aA[idx];
        wchar_t &p5 = (wchar_t &)aA[idx];

        aA[idx] = 0;
    });

    A = aA;

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

