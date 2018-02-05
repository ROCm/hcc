// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test checks that the addition operator works inside a vector function</summary>

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
void InitializeArray(vector<float> &vM, int size)
{
    for(int i=0; i<size; ++i)
    {
        vM[i] = rand();
    }
}

// Vector function for testing the addition operator
void kernel(index<2> idx, array<float, 2> &aC, array<float, 2> &aA, array<float, 2> &aB) __GPU
{
    // Addition
    aC[idx] = aA[idx] + aB[idx];
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
    vector<float> A(size);
    vector<float> B(size);

    // Initialize input
    srand(1997);
    InitializeArray(A, size);
    InitializeArray(B, size);

    // --Start defining GPU workload --
    // Out computation follows a 2D shaped extent of size N x M
    extent<2> e(N, M);

    // setup input arrays
    array<float, 2> aA(e, A.begin(), A.end(), rv), aB(e, B.begin(), B.end(), rv);

    // setup output
    array<float, 2> aC(e, rv);
    vector<float> C(size);

    parallel_for_each(aA.get_extent(), [&](index<2> idx) __GPU {
        kernel(idx, aC, aA, aB);
    });

    C = aC;

    bool passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            float expectedPc = A[i * N + j] + B[i * N + j];

            if (fabs(C[i * N + j] - expectedPc) > FLT_EPSILON)
            {
                printf("\nAddition aails\n");
                printf("-Expression: %f + %f", A[i * N + j], B[i * N + j]);
                printf("-Actual C[%d]: %f, ExpectedPc: %f\n", i * N + j, C[i * N + j], expectedPc);
                passed = false;
                break;
            }
        }
    }

    printf("%s: %s\n",  argv[0], passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

