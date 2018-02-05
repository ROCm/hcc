// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test check that the built-in logical OR operator works inside a vector function</summary>

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
void InitializeArray(vector<long> &vM, int size)
{
    for(int i=0; i<size; ++i)
    {
        // Set half of the values to 0, rest to non-zero
        int randomVal = rand();
        vM[i] = (randomVal < RAND_MAX/2) ? 0 : randomVal;
    }
}

// Vector function testing the logical OR operator, ||
void Kernel(index<2> idx, array<long, 2> &aC, array<long, 2> &aA, array<long, 2> &aB) __GPU
{
    // Logical OR, ||
    aC[idx] = aA[idx] || aB[idx];
}

// Main entry point
int main(int argc, char **argv)
{

    accelerator device;
    if(!get_device(Device::ALL_DEVICES, device))
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
    vector<long> A(size);
    vector<long> B(size);

    // Initialize input
    srand(3163);
    InitializeArray(A, size);
    InitializeArray(B, size);

    // --Start defining GPU workload --
    // Out computation follows a 2D shaped extent of size N x M
    extent<2> e(N, M);

    // setup input arrays
    array<long, 2> aA(e, A.begin(), A.end(), rv), aB(e, B.begin(), B.end(), rv);

    // setup output
    array<long, 2> aC(e, rv);
    vector<long> C(size);

    parallel_for_each(aA.get_extent(), [&](index<2> idx) __GPU {
        Kernel(idx, aC, aA, aB);
    });

    C = aC;

    // Check GPU results
    bool passed = true;
    for (int i=0; i<M &&passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            long expectedPc = A[i * N + j] || B[i * N + j];

            if (C[i * N + j] != expectedPc)
            {
                printf("\nLogical OR aails\n");
                printf("-Expression: %ld || %ld", A[i * N + j] , B[i * N + j]);
                printf("-Actual C[%d]: %ld, Expected: %ld\n", i * N + j, C[i * N + j], expectedPc);
                passed = false;
                break;
            }
        }
    }

    printf("%s: %s\n",  argv[0], passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}
