// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>(Use reference instead) Define pointers which point to const data. Verify the value of the data can be got through the pointers. </summary>

#include "../../inc/common.h"

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
        s o;

        o.i = 2;
        o.d = 2;
        o.ul = 2;
        o.f = 2;

        const s o2 = o;

        const s &ps = o2;

        s o3 = ps;

        if (!Equal(o3.i, 2) || !Equal(o3.d, (double)2) || !Equal(o3.ul, (unsigned long)2) || !Equal(o3.f, (float)2))
            aA[idx] = 1;

        const bool b1 = true;
        const bool &pb1 = b1;
        bool b2 = pb1;

        const int i1 = 1;
        const int &pi1 = i1;
        int i2 = pi1;

        const double d1 = 1;
        const double &pd1 = d1;
        double d2 = pd1;

        if (!Equal(b2, b1) || !Equal(i2, i1) || !Equal(d2, d1))
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

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}


