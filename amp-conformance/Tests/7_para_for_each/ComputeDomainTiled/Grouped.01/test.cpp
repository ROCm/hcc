//--------------------------------------------------------------------------------------
// File: test.cpp
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//
/// <tags>P1</tags>
/// <summary>Irregular shapes of groups, additionally make sure that A_x is dividable by D_x , but not by D_y and D_z, similarly for A_y and A_z they should be dividable only by their corresponding group size value.</summary>

#include <amptest.h>
#include <vector>

using std::vector;
using namespace Concurrency;

static const unsigned int M = 2 * 2, N = 3 * 3, W = 7 * 7;
static const unsigned int size = M * N * W;

void kernel1(int &c, int a, int b) __GPU
{
    int one = a - a + b/(-a + b + a);
    c = one * (a + b);
}

void vector_invoker(const vector<int> &A, const vector<int> &B, vector<int> &C, accelerator_view &av)
{
    extent<1> vector(size);
    const array<int, 1> fA(vector, A.begin(), A.end(), av);
    const array<int, 1> fB(vector, B.begin(), B.end(), av);
    array<int, 1> fC(vector, av);

    parallel_for_each(vector.tile<7>(), [&] (tiled_index<7> ti) __GPU
    {
        kernel1(fC[ti], fA[ti], fB[ti]);
    });

    C = fC;
}

void kernel2(index<2> idx, int &c, const array<int, 2> &fA, int b) __GPU
{
    c = fA[idx] + b;
}

void matrix_invoker(const vector<int> &A, const vector<int> &B, vector<int> &C, accelerator_view &av)
{
    extent<2> matrix(size/W, W);
    array<int, 2> fA(matrix, A.begin(), A.end(), av);
    array<int, 2> fB(matrix, B.begin(), B.end(), av);
    array<int, 2> fC(matrix, av);

    parallel_for_each(matrix.tile<N, W>(), [&] (tiled_index<N, W> ti) __GPU
    {
        kernel2(ti, fC[ti], fA, fB[ti]);
    });

    C = fC;
}

void kernel3(tiled_index<2, 3, 7> ti, array<int, 3> &fC, int a, int b) __GPU
{
    fC[ti] = a + b;
}

void cube_invoker(const vector<int> &A, const vector<int> &B, vector<int> &C, accelerator_view &av)
{
    extent<3> cube(M, N, W);
    array<int, 3> fA(cube, A.begin(), A.end(), av);
    array<int, 3> fB(cube, B.begin(), B.end(), av);
    array<int, 3> fC(cube, av);

    parallel_for_each(cube.tile<2, 3, 7>(), [&] (tiled_index<2, 3, 7> ti) __GPU
    {
        kernel3(ti, fC, fA[ti], fB[ti]);
    });

    C = fC;
}

int int_add_grouped(void(*invoker)(const vector<int> &, const vector<int> &, vector<int> &, accelerator_view&), const char *version)
{
    vector<int> A(size);
    vector<int> B(size);
    vector<int> C(size);
    vector<int> refC(size);

    for (unsigned int i = 0; i < size; i++)
    {
        A[i] = i;
        B[i] = i;
    }

    for (unsigned int i = 0; i < size; i++)
    {
        refC[i] = A[i] + B[i];
    }

    accelerator device;
    if (!Test::get_device(Test::Device::ALL_DEVICES, device))
    {
        std::cout << "Unable to get requested accelerator" << std::endl;
        return 2;
    }
    accelerator_view av = device.get_default_view();

    invoker(A, B, C, av);

    bool passed = Test::Verify(C, refC);

    printf("int_add_grouped_%s: %s\n", version, passed ? "Passed!" : "Failed!");

    return !passed;
}


int main()
{
    int status = 0;

    status = int_add_grouped(vector_invoker, "vector");
    if (status)
    {
        return status;
    }

    status = int_add_grouped(matrix_invoker, "matrix");
    if (status)
    {
        return status;
    }

    status = int_add_grouped(cube_invoker, "cube");

    return status;
}

