// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>(Negative) Verify const data cannot be changed through pointer.</summary>
//#Expects: Error: test.cpp\(53\) : error C3490:.*(\bi\b).*(cannot be modified because it is being accessed through a const object)?
//#Expects: Error: test.cpp\(54\) : error C3490:.*(\bd\b).*(cannot be modified because it is being accessed through a const object)?
//#Expects: Error: test.cpp\(55\) : error C3490:.*(\bul\b).*(cannot be modified because it is being accessed through a const object)?
//#Expects: Error: test.cpp\(56\) : error C3490:.*(\bf\b).*(cannot be modified because it is being accessed through a const object)?
//#Expects: Error: test.cpp\(59\) : error C3892:.*(\bpi1\b).*:.*(you cannot assign to a variable that is const)?
//#Expects: Error: test.cpp\(62\) : error C3892:.*(\bpd1\b).*:.*(you cannot assign to a variable that is const)?

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
    vector<s> G(size);
    vector<int> Gi(size);
    vector<double> Gd(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);
    array<s, 1> aG(e, G.begin(), rv);
    array<int, 1> aGi(e, Gi.begin(), rv);
    array<double, 1> aGd(e, Gd.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        const s o2 = aG[idx];

        const s* ps = &o2;

        ps->i = 1; // not allowed here
        ps->d = 1;
        ps->ul = 1;
        ps->f = 1;

        const int *pi1 = &aGi[idx];
        *pi1 = 2;
        const double d1 = 1;
        const double *pd1 = &aGd[idx];
        *pd1 = 2.0;

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

