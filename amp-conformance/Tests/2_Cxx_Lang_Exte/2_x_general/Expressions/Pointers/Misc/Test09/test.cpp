// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Define a class which has array_view. Define pointers to that struct. Verify the contents of array_view can be changed..</summary>

#include "../../inc/common.h"

class c1
{
public:
    c1(array_view<int, 1> a) __GPU : m_av(a)   {}
    array_view<int, 1> m_av;
    void add(array_view<int, 1> av, index<1>idx) __GPU
    {
        m_av[idx] += av[idx];
    }
    void add2(array_view<int, 1> &av, index<1>idx) __GPU
    {
        m_av[idx] += av[idx];
    }

    ~c1() __GPU {}
};

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
    array<int, 1> aG(e, G.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1>idx) __GPU
    {
        array_view<int, 1> av(aG);

        c1 o(av);

        c1 *p = &o;

        p->add(av, idx);

        if (aG[idx] != 2)
            aA[idx] = 1;

        p->add2(av, idx);

        if (aG[idx] != 4)
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

    passed = test(rv);

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}
