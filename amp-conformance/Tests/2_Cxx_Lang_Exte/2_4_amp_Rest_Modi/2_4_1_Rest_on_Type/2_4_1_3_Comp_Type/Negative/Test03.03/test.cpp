// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>(Negative) Define pointer arrays whose pointers point to amp–compatible type and non-amp–compatible type.</summary>
//#Expects: Error: test.cpp\(50\) : error C3581:.*(\bc1 \*\[1\]).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(51\) : error C3581:.*(\bc2 \*\[2\]).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(52\) : error C3581:.*(\bs1 \*\[3\]).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(53\) : error C3581:.*(\bu1 \*\[4\]).*:.*(unsupported type in amp restricted code)?

#include <amptest.h>
#include <vector>

using std::vector;
using namespace Concurrency;

#define BLOCK_DIM 16

class c1
{
public:
    wchar_t c;
};

class c2
{
public:
    wchar_t c;
};

struct s1
{
    short int si;
};

union u1
{
    long double ud;
    int i;
};

struct FunctObj
{
    FunctObj(array<int, 2> &fA, array<int, 2> &fB, array<int, 2> &fC):mA(fA), mB(fB), mC(fC) {}

    void operator()(tiled_index<BLOCK_DIM, BLOCK_DIM> idx) __GPU_ONLY
    {
        c1 *p1[1]; // not allowed here
        c2 *p2[2];
        s1 *p3[3];
        u1 *p4[4];
    }

private:
    array<int, 2> &mA;
    array<int, 2> &mB;
    array<int, 2> &mC;
};

runall_result test_main()
{
    srand(2009);
    const int M = 256;

    vector<int> A(M * M);
    vector<int> B(M * M);
    vector<int> C(M * M);
    vector<int> refC(M * M);

    for (size_t i = 0; i < M * M; i++)
    {
        A[i] = rand();
    }

    for (size_t i = 0; i < M * M; i++)
    {
        B[i] = rand();
    }

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            refC[i * M + j] += A[i * M + j] + B[i * M + j];
        }
    }

    accelerator device = require_device_with_double(Test::Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    extent<2> e(M, M);

    array<int, 2> fA(e, A.begin(), rv);
    array<int, 2> fB(e, B.begin(), rv);
    array<int, 2> fC(e, rv);

    FunctObj cobj(fA, fB, fC);

    parallel_for_each(e.tile<BLOCK_DIM, BLOCK_DIM>(), cobj);

    C = fC;

    bool passed = true;
    for(size_t i = 0; i < M * M; i++)
    {
        if (refC[i] * 4 != C[i])
        {
            printf("C[%zu] = %d, refC[%zu] = %d\n", i, C[i], i, refC[i]);
            passed = false;
            break;
        }
    }

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return runall_fail;
}

