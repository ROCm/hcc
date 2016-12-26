// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>For e1 && e2 && e3, set e1 to true and e2 to false. Verify e3 is not evaluated.</summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// this tests uses a 2D dataset of size N x M
const int N = 1024;
const int M = 512;
const int size = N * M;
const int DEFAULT = 1;

void kernel_if(index<2> idx, array<int, 2> &aA, int e1, int e2) __GPU
{
    int i = DEFAULT;
    if (e1 && e2 && (++i))
    {
        aA[idx] = 3;
    }

    aA[idx] = i;
}

void kernel_switch(index<2> idx, array<int, 2> &aA, int e1, int e2) __GPU
{
    int i = DEFAULT;
    switch (e1 && e2 && (++i))
    {
    case false:
        aA[idx] = 3;
        break;
    }

    aA[idx] = i;
}

void kernel_whiledo(index<2> idx, array<int, 2> &aA, int e1, int e2) __GPU
{
    int i = DEFAULT;
    while (e1 && e2 && (++i))
    {
        aA[idx] = 3;
        break;
    }

    aA[idx] = i;
}

void kernel_dowhile(index<2> idx, array<int, 2> &aA, int e1, int e2) __GPU
{
    int i = DEFAULT;
    int j = 0;
    do
    {
        j++;
        if (j > 1)
            break;
    } while (e1 && e2 && (++i));

    aA[idx] = i;
}

void kernel_for(index<2> idx, array<int, 2> &aA, int e1, int e2) __GPU
{
    int i = DEFAULT;

    for (;(e1 && e2 && (++i));)
    {
        aA[idx] = 3;
    }

    aA[idx] = i;
}

bool verify(vector<int> &v, array<int, 2> &aA)
{
    v = aA;

    for (int i = 0; i < size; i++)
    {
        if (v[i] != DEFAULT)
            return false;
    }

    return true;
}

// Main entry point
runall_result test_main()
{
    accelerator_view rv =  require_device(Device::ALL_DEVICES).get_default_view();

    vector<int> A(size);
    vector<int> Zero(size);

    for (int i = 0; i < M * N; i++)
    {
        A[i] = 0;
        Zero[i] = 0;
    }

    extent<2> e(N, M);

    // setup input arrays
    array<int, 2> aA(e, A.begin(), A.end(), rv);

    int e1 = 1;
    int e2 = 0;

    parallel_for_each(aA.get_extent(), [&, e1, e2](index<2> idx) __GPU {
        kernel_if(idx, aA, e1, e2);
    });

    if (!verify(A, aA))
    {
        printf("failed\n");
        return runall_fail;
    }

    copy(Zero.begin(), Zero.end(), aA);

    parallel_for_each(aA.get_extent(), [&, e1, e2](index<2> idx) __GPU {
        kernel_switch(idx, aA, e1, e2);
    });

    if (!verify(A, aA))
    {
        printf("failed\n");
        return runall_fail;
    }

    copy(Zero.begin(), Zero.end(), aA);

    parallel_for_each(aA.get_extent(), [&, e1, e2](index<2> idx) __GPU {
        kernel_whiledo(idx, aA, e1, e2);
    });

    if (!verify(A, aA))
    {
        printf("failed\n");
        return runall_fail;
    }

    copy(Zero.begin(), Zero.end(), aA);

    parallel_for_each(aA.get_extent(), [&, e1, e2](index<2> idx) __GPU {
        kernel_dowhile(idx, aA, e1, e2);
    });

    if (!verify(A, aA))
    {
        printf("failed\n");
        return runall_fail;
    }


    copy(Zero.begin(), Zero.end(), aA);

    parallel_for_each(aA.get_extent(), [&, e1, e2](index<2> idx) __GPU {
        kernel_for(idx, aA, e1, e2);
    });

    if (!verify(A, aA))
    {
        printf("failed\n");
        return runall_fail;
    }

    printf("passed\n");
    return runall_pass;
}

