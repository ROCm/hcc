// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test the compound type. Change the value </summary>

#include "../../inc/common.h"

class derived_c
{
public:
    void f1(double d) {m_d = d;}
    float f21() {return m_f;}
    void f1(int i) {m_i = i;}
    unsigned long f2() {return m_ul;}

    int m_i;
    unsigned long m_ul;
    float m_f;
    double m_d;
};

struct derived_s
{
public:
    void f1(double d) {m_d = d;}
    float f21() {return m_f;}
    void f1(int i) {m_i = i;}
    unsigned long f2() {return m_ul;}

    int m_i;
    unsigned long m_ul;
    float m_f;
    double m_d;
};

template <typename type>
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

    parallel_for_each(aA.get_extent().tile<1>(), [&](tiled_index<1>idx) __GPU_ONLY
    {
        tile_static type v1;

        v1.m_i = 1;
        v1.m_ul = 1;
        v1.m_f = 1;
        v1.m_d = 1;

        type *p = &v1;

        p->m_i = 2;
        p->m_ul = 2;
        p->m_f = 2;
        p->m_d = 2;

        if (!Equal(v1.m_i, (int)2) || !Equal(v1.m_ul, (unsigned long)2) || !Equal(v1.m_f, (float)2) || !Equal(v1.m_d, (double)2))
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
    passed = test<derived_c>(rv) && test<derived_s>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

