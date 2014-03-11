// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Neg: The pointers are members of structures, classes, union of amp-compatible type. Compilation fails.</summary>
//#Expects: Error: test.cpp\(59\) : error C3581:.*(\bc1\b).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(60\) : error C3581:.*(\bs1\b).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(61\) : error C3581:.*(\bu1\b).*:.*(unsupported type in amp restricted code)?

#include "common.h"

class c1
{
public:
    int *pi;
    double *pd;
    unsigned long *pul;
    float *pf;
};

struct s1
{
public:
    int *pi;
    double *pd;
    unsigned long *pul;
    float *pf;
};

union u1
{
public:
    int *pi;
    double *pd;
    unsigned long *pul;
    float *pf;
};

bool test(accelerator_view &rv)
{

    const int size = 100;

    vector<int> A(size);

    for(int i = 0; i < size; i++)
    {
        A[i] = INIT_VALUE;
    }

    extent<1> e(size);

    array<int, 1> aA(e, A.begin(), rv);

    parallel_for_each(aA.get_extent().tile<1>(), [&](tiled_index<1>idx) __GPU
    {
        tile_static c1 p1; // classes which have pointers are not allowed here
        tile_static s1 p2;
        tile_static u1 p3;

        aA[idx] = 1;
    });

    A = aA;

    for (int i =  0; i < size; i++)
    {
        if (A[i] != INIT_VALUE)
            return false;
    }

    return true;
}

runall_result test_main(int argc, char *argv)
{
    bool passed = true;

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    passed = test(rv);
    passed = false;

    printf("%s\n", passed ? "Passed!" : "Failed!");

    return runall_fail;
}

