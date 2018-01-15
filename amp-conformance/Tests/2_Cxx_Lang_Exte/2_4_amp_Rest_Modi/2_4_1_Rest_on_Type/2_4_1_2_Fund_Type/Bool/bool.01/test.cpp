// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test the use of bool</summary>

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

bool GreaterThan(int a, int b) __GPU
{
    return a > b;
}

bool LessThan(int a, int b) __GPU
{
    return a < b;
}

void kernel1( int & c,  int & d,  int & e,  int & f,  int & g,  int & h, int a, int b) __GPU
{
    bool mc = (a == b);
    bool md = (a >= b);
    bool me = GreaterThan(a, b);
    bool mf = (a <= b);

    c = (mc == true) ? 1 : 0;
    d = (md == true) ? 1 : 0;
    e = (me == true) ? 1 : 0;
    f = (mf == true) ? 1 : 0;

    bool mg = LessThan(a, b);
    g = (mg == true) ? 1 : 0;
    bool mh = (a != b);
    // implicit conversion
    h = mh;
}

int test1(accelerator_view &rv)
{
    const int N = 1024;

    const int size = N;

    // Input datasets
    vector<int> A(size);
    vector<int> B(size);

    // Initialize input
    srand(25763);
    InitializeArray(A, size);
    InitializeArray(B, size);

    // --Start defining GPU workload --
    extent<1> e(N);

    // setup input arrays
    array<int, 1> aA(e, A.begin(), A.end(), rv), aB(e, B.begin(), B.end(), rv);

    // setup output
    array<int, 1> aC(e, rv), aD(e, rv), aE(e, rv), aF(e, rv), aG(e, rv), aH(e, rv);
    vector<int> C(size);
    vector<int> D(size);
    vector<int> E(size);
    vector<int> F(size);
    vector<int> G(size);
    vector<int> H(size);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU {
        kernel1(aC[idx], aD[idx], aE[idx], aF[idx], aG[idx], aH[idx], aA[idx], aB[idx]);
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
    for (int i=0; i<N && passed; ++i)
    {
        int expectedPc = (A[i] == B[i]) ? 1 : 0;

        if (C[i] != expectedPc)
        {
            printf("\nEquality operator test failed\n");
            printf("-Expression: %d == %d\n", A[i], B[i]);
            printf("-Actual C[%d]: %d, ExpectedPc: %d\n", i, C[i], expectedPc);
            passed = false;
            numFail++;
            break;
        }
    }

    // Greater than or equal to, >=
    passed = true;
    for (int i=0; i<N && passed; ++i)
    {
        int expectedPd = (A[i] >= B[i]) ? 1 : 0;

        if (D[i] != expectedPd)
        {
            printf("\nGreater than or equal to operator failed\n");
            printf("-Expression: %d >= %d\n", A[i], B[i]);
            printf("-Actual D[%d]: %d, ExpectedPd: %d\n", i, D[i], expectedPd);
            passed = false;
            numFail++;
            break;
        }
    }

    // Greater than, >
    passed = true;
    for (int i=0; i<N && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPe = (A[i] > B[i]) ? 1 : 0;

            if (E[i] != expectedPe)
            {
                printf("\nGreater than operator failed\n");
                printf("-Expression: %d > %d\n", A[i], B[i]);
                printf("-Actual E[%d]: %d, ExpectedPe: %d\n", i, E[i], expectedPe);
                passed = false;
                numFail++;
                break;
            }
        }
    }

    // Less than or equal to, <=
    passed = true;
    for (int i=0; i<N && passed; ++i)
    {
        int expectedPf = (A[i] <= B[i]) ? 1 : 0;

        if (F[i] != expectedPf)
        {
            printf("\nLess than or equal to operator failed\n");
            printf("-Expression: %d <= %d", A[i], B[i]);
            printf("-Actual F[%d]: %d, ExpectedPf: %d\n", i, F[i], expectedPf);
            passed = false;
            numFail++;
            break;
        }
    }

    // Less than, <
    passed = true;
    for (int i=0; i<N && passed; ++i)
    {
        int expectedPg = (A[i] < B[i]) ? 1 : 0;

        if (G[i] != expectedPg)
        {
            printf("\nLess than operator failed\n");
            printf("-Expression: %d < %d\n", A[i], B[i]);
            printf("-Actual G[%d]: %d, ExpectedPg: %d\n", i, G[i], expectedPg);
            passed = false;
            numFail++;
            break;
        }
    }

    // Not equal to, !=
    passed = true;
    for (int i=0; i<N && passed; ++i)
    {
        int expectedPh = (A[i] != B[i]) ? 1 : 0;

        if (H[i] != expectedPh)
        {
            printf("\nLess than operator failed\n");
            printf("-Expression: %d < %d\n", A[i], B[i]);
            printf("-Actual H[%d]: %d, ExpectedPh: %d\n", i, H[i], expectedPh);
            passed = false;
            numFail++;
            break;
        }
    }

    if(numFail > 0)
    {
        printf("\ntest1: %d test(s) failed\n", numFail);
    }
    else
    {
        printf("\ntest1: all tests passed\n");
    }

    return numFail > 0 ? 1 : 0;
}

#define BOUND1  100
#define BOUND2  1000
#define BOUND3  5000
#define BOUND4  12000
#define BOUND5  20000
#define BOUND6  30000

void kernel2( int & c, int a) __GPU
{
    bool m1 = (a > BOUND1);
    bool m2 = (a < BOUND2);
    bool m3 = (a > BOUND3);
    bool m4 = (a < BOUND4);
    bool m5 = (a > BOUND5);
    bool m6 = (a < BOUND6);

    bool result = (m1 && m2) || (m3 && m4) || (m5 && m6);
    c = result ? 1 : 0;
}

int test2(accelerator_view &rv)
{
    const int N = 1024;

    const int size = N;

    // Input datasets
    vector<int> A(size);

    // Initialize input
    srand(2010);
    InitializeArray(A, size);

    // --Start defining GPU workload --
    extent<1> e(N);

    // setup input arrays
    array<int, 1> aA(e, A.begin(), A.end(), rv);

    // setup output
    array<int, 1> aC(e, rv);
    vector<int> C(size);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU {
        kernel2(aC[idx], aA[idx]);
    });

    C = aC;

    bool passed = true;
    for (int i=0; i<N; ++i)
    {
        int a = A[i];
        int expectedPc = ((a > BOUND1) && (a < BOUND2)) ||
            ((a > BOUND3) && (a < BOUND4)) ||
            ((a > BOUND5) && (a < BOUND6));

        if (C[i] != expectedPc)
        {
            printf("\nEquality operator test failed\n");
            printf("-Actual C[%d]: %d, ExpectedPc: %d\n", i, C[i], expectedPc);
            passed = false;
            break;
        }
    }

    if(passed == false)
    {
        printf("\ntest2: test failed\n");
    }
    else
    {
        printf("\ntest2: test passed\n");
    }

    return passed ? 0 : 1;
}

int main()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    return test1(rv) || test2(rv);
}
