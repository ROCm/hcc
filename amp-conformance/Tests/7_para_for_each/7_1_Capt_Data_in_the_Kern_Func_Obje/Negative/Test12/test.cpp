// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Neg: Capture other pointers by value.</summary>
//#Expects: Error: test.cpp\(33\) : error C3596:.*(\bp\b).*(\bint \*).*:.*(variable captured by lambda has unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(33\) : error C3581:.*(\btest::<lambda_\w*>).*:.*(unsupported type in amp restricted code)?

#include "../common.h"

template <typename type>
bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);
    vector<int> G(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        G[i] = 1;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    type *p = NULL;
    parallel_for_each(aA.get_extent(), [&, p](index<1>idx) __GPU
    {
        type *p1 = p; // not allowed here
        aA[idx] = 1;
    });

    A = aA;

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

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return runall_skip;
    }

    accelerator_view rv = device.get_default_view();

    passed = test<int>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

