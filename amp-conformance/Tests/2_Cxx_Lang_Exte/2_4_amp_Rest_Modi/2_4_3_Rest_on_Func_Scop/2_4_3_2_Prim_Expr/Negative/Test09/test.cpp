// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Neg: The pointer points static member. Compilation fails</summary>
//#Expects: Error: test.cpp\(35\) : error C3586:.*(\bi\b).*:.*(using global or static variables is unsupported in amp restricted code)?

#include "common.h"

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

    static int i = 0;
    static unsigned int ui = 0;
    static float f = 0;
    static double d = 0;

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {

        int *pi = &i; // not allowed here
        int &ri = i;

        unsigned int *pui = &ui;
        unsigned int &rui = ui;

        float *pf = &f;
        float &rf = f;

        double *pd = &d;
        double rd = d;

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

