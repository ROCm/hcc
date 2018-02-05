// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test the compound type. Change the value </summary>

#include "../../inc/common.h"

class base_c
{
public:
    void f1(int i) {m_i = i;}
    unsigned long f2() {return m_ul;}

    int m_i;
    unsigned long m_ul;
};

class derived_c : public base_c
{
public:
    void f1(double d) {m_d = d;}
    float f21() {return m_f;}

    float m_f;
    double m_d;
};

struct base_s
{
public:
    void f1(int i) {m_i = i;}
    unsigned long f2() {return m_ul;}

    int m_i;
    unsigned long m_ul;
};

struct derived_s : public base_s
{
public:
    void f1(double d) {m_d = d;}
    float f21() {return m_f;}

    float m_f;
    double m_d;
};

union u
{
public:
    void f1(int i) {m_i = i;}
    unsigned long f2() {return m_ul;}
    void f1(double d) {m_d = d;}
    float f21() {return m_f;}

    float m_f;
    double m_d;
    int m_i;
    unsigned long m_ul;
};

template <typename type>
bool VerifyR(type in, int in2, double d2) __GPU
{
    if (!Equal(in.m_i, (int)in2) || !Equal(in.m_ul, (unsigned long)in2) || !Equal(in.m_f, (float)in2) || !Equal(in.m_d, d2))
        return false;

    return true;
}

template <>
bool VerifyR<u>(u in, int in2, double d2) __GPU
{
    if (!Equal(in.m_d, d2))
        return false;

    return true;
}


template <typename type>
bool test(accelerator_view &rv)
{
    int data[] = {0, 0, 0, 0};
    vector<int> Flags(data, data + sizeof(data) / sizeof(int));
    extent<1> eflags(sizeof(data) / sizeof(int));
    array<int, 1> aFlag(eflags, Flags.begin(), rv);

    const int size = 100;

    vector<type> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i].m_i = 1;
        A[i].m_ul = 1;
        A[i].m_f = 1.0f;
        A[i].m_d = 1.0;
    }

    extent<1> e(size);

    array<type, 1> aA(e, A.begin(), rv), aA2(e, rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        type *p = NULL;

        if (aFlag[0] == 0)
            p =&aA[idx];
        else
            p =&aA2[idx];

        p->m_i = 2;
        p->m_ul = 2;
        p->m_f = 2.0f;
        p->m_d = 2.0;

    });

    A = aA;

    for (int i =  0; i < size; i++)
    {
        if (!VerifyR(A[i], 2, 2.0))
            return false;
    }

    return true;
}

runall_result test_main()
{
    bool passed = true;

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test<derived_c>(rv) && test<derived_s>(rv) && test<u>(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

