// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test check that the built-in multiplication operator works inside a vector function</summary>

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// Initialize the input with random data
void InitializeArray(vector<unsigned int> &vM, int size)
{
    for(int i=0; i<size; ++i)
    {
        // Shift range from [0, RAND_MAX] to [1, RAND_MAX + 1]
        vM[i] = rand() + 1;
    }
}

// Vector function testing the multiplication operators, *
void kernel(index<2> idx, array<unsigned int, 2> &aC, array<unsigned int, 2> &aA, array<unsigned int, 2> &aB) __GPU
{
    aC[idx] = aA[idx] * aB[idx];
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
    vector<unsigned int> A(size);
    vector<unsigned int> B(size);

    // Initialize input
    srand(31271);
    InitializeArray(A, size);
    InitializeArray(B, size);

    // --Start defining GPU workload --
    // Out computation follows a 2D shaped extent of size N x M
    extent<2> e(N, M);

    // setup input arrays
    array<unsigned int, 2> aA(e, A.begin(), A.end(), rv), aB(e, B.begin(), B.end(), rv);

    // setup output
    array<unsigned int, 2> aC(e, rv);
    vector<unsigned int> C(size);

    parallel_for_each(aA.get_extent(), [&](index<2> idx) __GPU {
        kernel(idx, aC, aA, aB);
    });

    C = aC;

    // Check GPU results
    unsigned int passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            unsigned int expectedPc = A[i * N + j] * B[i * N + j];

            if (C[i * N + j] != expectedPc)
            {
                printf("\nMultiplication aails\n");
                printf("-Expression: %d * %d\n", A[i * N + j], B[i * N + j]);
                printf("-Actual C[%d]: %d, ExpectedPc: %d\n", i * N + j, C[i * N + j], expectedPc);
                passed = false;
                break;
            }
        }
    }

    printf("%s: %s\n",  argv[0], passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}
