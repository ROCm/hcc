// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test checks that the built-in Bitwise AND assignment operator works inside a vector function</summary>

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// Initialize the input with random data
void InitializeArray(vector <int> &vM, int size, int value = 0)
{
    // zero not allowed because we use it for division
    if(value == 0) value = rand() + 1;

    for(int i=0; i<size; ++i)
    {
        vM[i] = value;
    }
}

// This kernel tests the bitwise AND assignment operator, &=
void kernel(index<2> idx, array<int, 2> &aB, array<int, 2> &aA) __GPU // input
{
    aB[idx] &= aA[idx];
}

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
    const int N = 4;
    const int M = 4;
    const int size = M * N;
    extent<2> e(N, M);

    // Initialize input, outputs
    srand(13);

    vector<int> A(size);
    InitializeArray(A, size);
    array<int, 2> aA(e, A.begin(), A.end(), rv);

    // these are both input & output arrays
    vector<int> B(size);
    int value = rand();
    InitializeArray(B, size, value);
    array<int, 2> aB(e, B.begin(), B.end(), rv);

    parallel_for_each(aA.get_extent(), [&](index<2> idx) __GPU {
        kernel(idx, aB, aA);
    });

    B = aB;

    // Verify results

    bool passed = true;
    for (int i=0; i<M && passed; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            int expectedPb = value;
            expectedPb &= A[i * N + j];

            if (B[i * N + j] != expectedPb)
            {
                printf("\nBitwise AND Assignment failed\n");
                printf("-Expression: %d &= %d\n", value, A[i * N + j]);
                printf("-Actual: B[%d] = %d, Expected: %d\n", i * N + j, B[i * N + j], expectedPb);
                passed = false;
                break;
            }
        }
    }

    printf("%s: %s\n",  argv[0], passed ? "Passed!" : "Failed!");

    return passed ? 0 : 1;
}

