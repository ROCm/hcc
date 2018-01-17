// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test check that built-in relational and equality operators work inside a vector function</summary>

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <float.h>
#include <time.h>
#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// Initialize the input with random data
void InitializeArray(vector <int> &vM, int size)
{
    for(int i=0; i<size; ++i)
    {
        vM[i] = rand();
    }
}

// Vector function testing the additive operators
void KernelWithRelationalOperators(index<2> idx,
    array<int, 2> &aC, array<int, 2> &aD, array<int, 2> &aE, array<int, 2> &aF, array<int, 2> &aG, array<int, 2> &aH,
    array<int, 2> &aA, array<int, 2> &aB) __GPU
{
    // Equality, ==
    bool mc = (aA[idx] == aB[idx]);
    if(mc == true)
    {
        aC[idx] = 1;
    }
    else
    {
        aC[idx] = 0;
    }

    // Greater than or equal to, >=
    bool md = (aA[idx] >= aB[idx]);
    if(md == true)
    {
        aD[idx] = 1;
    }
    else
    {
        aD[idx] = 0;
    }

    // Greater than, >
    bool me = (aA[idx] > aB[idx]);
    if(me == true)
    {
        aE[idx] = 1;
    }
    else
    {
        aE[idx] = 0;
    }

    // Less than or equal to, <=
    bool mf = (aA[idx] <= aB[idx]);
    if(mf == true)
    {
        aF[idx] = 1;
    }
    else
    {
        aF[idx] = 0;
    }

    // Less than, <
    bool mg = (aA[idx] < aB[idx]);
    if(mg == true)
    {
        aG[idx] = 1;
    }
    else
    {
        aG[idx] = 0;
    }

    // Not equal to, !=
    bool mh = (aA[idx] != aB[idx]);
    if(mh == true)
    {
        aH[idx] = 1;
    }
    else
    {
        aH[idx] = 0;
    }
}

// Main entry point
int main(int argc, char **argv)
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    // this tests uses a 2D dataset of size N x M
    const int N = 1024;
    const int M = 512;

    const int size = N * M;

    // Input datasets
    vector<int> A(size);
    vector<int> B(size);

    // Initialize input
    srand(25763);
    InitializeArray(A, size);
    InitializeArray(B, size);

    // --Start defining GPU workload --
    // Out computation follows a 2D shaped extent of size N x M
    extent<2> e(N, M);

    // setup input arrays
    array<int, 2> aA(e, A.begin(), A.end(), rv), aB(e, B.begin(), B.end(), rv);

    // setup output
    array<int, 2> aC(e, rv), aD(e, rv), aE(e, rv), aF(e, rv), aG(e, rv), aH(e, rv);
    vector<int> C(size);
    vector<int> D(size);
    vector<int> E(size);
    vector<int> F(size);
    vector<int> G(size);
    vector<int> H(size);

    parallel_for_each(aA.get_extent(), [&](index<2> idx) __GPU {
        KernelWithRelationalOperators(idx, aC, aD, aE, aF, aG, aH, aA, aB);
    });

    C = aC;
    D = aD;
    E = aE;
    F = aF;
    G = aG;
    H = aH;

    // Check GPU results
    int numFail = 0;

    // Equality, ==
    bool passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPc = (A[i * N + j] == B[i * N + j]) ? 1 : 0;

            if (C[i * N + j] != expectedPc)
            {
                printf("\nEquality operator test failed\n");
                printf("-Expression: %d == %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual C[%d]: %d, ExpectedPc: %d\n", i * N + j, C[i * N + j], expectedPc);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Greater than or equal to, >=
    passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPd = (A[i * N + j] >= B[i * N + j]) ? 1 : 0;

            if (D[i * N + j] != expectedPd)
            {
                printf("\nGreater than or equal to operator failed\n");
                printf("-Expression: %d >= %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual D[%d]: %d, ExpectedPd: %d\n", i * N + j, D[i * N + j], expectedPd);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Greater than, >
    passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPe = (A[i * N + j] > B[i * N + j]) ? 1 : 0;

            if (E[i * N + j] != expectedPe)
            {
                printf("\nGreater than operator failed\n");
                printf("-Expression: %d > %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual E[%d]: %d, ExpectedPe: %d\n", i * N + j, E[i * N + j], expectedPe);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Less than or equal to, <=
    passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPf = (A[i * N + j] <= B[i * N + j]) ? 1 : 0;

            if (F[i * N + j] != expectedPf)
            {
                printf("\nLess than or equal to operator failed\n");
                printf("-Expression: %d <= %d", A[i * N + j], B[i * N + j]);
                printf("-Actual F[%d]: %d, ExpectedPf: %d\n", i * N + j, F[i * N + j], expectedPf);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Less than, <
    passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPg = (A[i * N + j] < B[i * N + j]) ? 1 : 0;

            if (G[i * N + j] != expectedPg)
            {
                printf("\nLess than operator failed\n");
                printf("-Expression: %d < %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual G[%d]: %d, ExpectedPg: %d\n", i * N + j, G[i * N + j], expectedPg);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Not equal to, !=
    passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPh = (A[i * N + j] != B[i * N + j]) ? 1 : 0;

            if (H[i * N + j] != expectedPh)
            {
                printf("\nLess than operator failed\n");
                printf("-Expression: %d < %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual H[%d]: %d, ExpectedPh: %d\n", i * N + j, H[i * N + j], expectedPh);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    if(numFail > 0)
    {
        printf("\n%s: %d test(s) failed\n", argv[0], numFail);
    }
    else
    {
        printf("\n%s: all tests passed\n", argv[0]);
    }

    return numFail > 0 ? 1 : 0;
}
