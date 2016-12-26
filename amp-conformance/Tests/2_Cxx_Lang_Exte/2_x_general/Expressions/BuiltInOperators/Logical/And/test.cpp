// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>This test check that the logical AND operator works inside a vector function</summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

// Vector function testing the logical AND operator
void Kernel(index<2> idx, array<long, 2> &aC, array<long, 2> &aA, array<long, 2> &aB) __GPU
{
    // Logical AND, &&
    aC[idx] = aA[idx] && aB[idx];
}

// Main entry point
runall_result test_main()
{
    accelerator_view rv = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    // this tests uses a 2D dataset of size N x M
    const int N = 1024;
    const int M = 512;

    const int size = N * M;

    // Input datasets
    vector<long> A(size);
    vector<long> B(size);

    // Initialize input
    srand(3163);
    Fill(A);
    Fill(B);

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
    runall_result result;
    for (int i=0; i<M &&result.get_is_pass(); ++i)
    {
        for(int j=0; j<N; ++j)
        {
            long expectedPc = A[i * N + j] && B[i * N + j];

            if (C[i * N + j] != expectedPc)
            {
                Log(LogType::Error, true) << "Logical AND fails" << std::endl;
                Log(LogType::Error, true) << "-Expression: " << A[i * N + j] << " && " << B[i * N + j] << std::endl;
                Log(LogType::Error, true) << "-Actual C[" << i * N + j << "]: " << C[i * N + j] << " Expected: " << expectedPc << std::endl;
                result = false;
                break;
            }
        }
    }

    return result;
}

