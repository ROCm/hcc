// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Test explicit pointer conversion.</summary>

#include "../../inc/common.h"

class c1
{
public:
    int i;
    double d;
    unsigned long ul;
    float f;
};

class c2 : public c1
{
public:
    uint32_t i2;
    double md;
};

template <typename T>
bool test(accelerator_view &rv)
{
    const int size = 100;

    vector<int> A(size);
    vector<c2> Gc(size);
    vector<int> Gi(size);
    vector<uint32_t> Gui(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
        Gc[i].i = 1;
        Gc[i].d = 1;
        Gc[i].ul = 1;
        Gc[i].f = 1;
        Gi[i] = 1;
        Gui[i] = 1;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);
    array<c2, 1> aGc(e, Gc.begin(), rv);
    array<int, 1> aGi(e, Gi.begin(), rv);
    array<uint32_t, 1> aGui(e, Gui.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        int *pi1 = &aGi[idx];
        bool *pb1 = (bool*)pi1;
        int i2 = *(int *)pb1;
        if (i2 != 1)
            aA[idx] = 1;

        uint32_t *pi2 = &aGui[idx];
        double * pd1 = (double *)pi2;
        uint32_t i4 = *(uint32_t *)pd1;

        if (i4 != 1)
            aA[idx] = 1;

        c1 o;
        o.i = 1;
        o.d = 1;
        o.ul = 1;
        o.f = 1;

        c2 *p = (c2 *)&aGc[idx];

        if (!Equal(p->i, (int)1) || !Equal(p->d, (double)1) || !Equal(p->f, (float)1) || !Equal(p->ul, (unsigned long)1))
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

    passed = test<int32_t>(rv) && test<unsigned long>(rv) && test<double>(rv) && test<float>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

