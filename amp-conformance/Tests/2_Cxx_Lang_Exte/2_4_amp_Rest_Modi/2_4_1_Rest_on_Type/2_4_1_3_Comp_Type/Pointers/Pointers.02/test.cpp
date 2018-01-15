// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test pointer comparison</summary>

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

void kernel1(index<1> idx, array<int, 1> &ac, array<int, 1> &aa, array<int, 1> &ab) __GPU
{
    index<1> idx0(0);

    int a = aa[idx];
    int b = ab[idx];

    int * pC = &ac[idx0];

    int * pC1 = pC + a;
    int * pC2 = pC + b;

    if (pC1 != pC2) {
        *(pC + idx[0]) = 1;
    } else {
        *(pC + idx[0]) = 2;
    }

}

void kernel2(index<1> idx, array<int, 1> &ac, array<int, 1> &aa, array<int, 1> &ab) __GPU
{
    index<1> idx0(0);

    int a = aa[idx];
    int b = ab[idx];

    int * pC = &ac[idx0];

    int * pC1 = pC + a;
    int * pC2 = pC + b;

    if (pC1 == pC2) {
        *(pC + idx[0]) = 2;
    } else {
        *(pC + idx[0]) = 1;
    }

}

void init_data(int size, vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &refC);
bool verify_result(int size, vector<int> &C, vector<int> &refC);

int int_vect_add()
{
    const size_t size = 1024;

    vector<int> A(size);
    vector<int> B(size);
    vector<int> C(size);
    vector<int> refC(size);

    init_data(size, A, B, C, refC);

    accelerator device;
    if (!Test::get_device(Test::Device::ALL_DEVICES, device))
    {
        std::cout << "Unable to get requested accelerator" << std::endl;
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    extent<1> vector(size);
    array<int, 1> aA(vector, A.begin(), A.end(), rv);
    array<int, 1> aB(vector, B.begin(), B.end(), rv);
    array<int, 1> aC(vector, C.begin(), C.end(), rv);
    array<int, 1> aD(vector, rv);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU
    {
        kernel1(idx, aC, aA, aB);
    });

    C = aC;

    bool passed;

    passed = verify_result(size, C, refC);

    printf("kernel1: %s\n", passed? "Passed!" : "Failed!");

    if (passed == false) goto Cleanup;

    C.assign(C.size(), 0);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) __GPU
    {
        kernel1(idx, aD, aA, aB);
    });

    C = aD;

    passed = verify_result(size, C, refC);

    printf("kernel2: %s\n", passed? "Passed!" : "Failed!");

Cleanup:

    return passed == true ? 0 : 1;
}

void init_data(int size, vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &refC)
{
    for (size_t i = 0; i < size; i++)
    {
        if (rand() % 2 == 0) {
            A[i] = static_cast<int>(i);
            B[i] = static_cast<int>(i);
        } else {
            A[i] = rand() % size;
            B[i] = rand() % size;
        }
    }

    for (size_t i = 0; i < size; i++)
    {
        refC[i] = (A[i] == B[i] ? 2 : 1);
    }
}

bool verify_result(int size, vector<int> &C, vector<int> &refC)
{
    bool passed = true;
    for(size_t i = 0; i < size; i++)
    {
        if (refC[i] != C[i])
        {
            fprintf(stderr, "C[%zu] = %d, refC[%zu] = %d\n", i, C[i], i, refC[i]);
            passed = false;
        }
    }

    return passed;
}

int main()
{
    return int_vect_add();
}

