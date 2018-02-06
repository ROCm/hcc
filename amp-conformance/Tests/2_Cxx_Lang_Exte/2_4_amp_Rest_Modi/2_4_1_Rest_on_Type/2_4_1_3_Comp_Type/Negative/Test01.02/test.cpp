// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Define pointers to non–amp–compatible type in kernel function </summary>
//#Expects: Error: test.cpp\(36\) : error C3581:.*(\bc1 \*).*:.*(unsupported type in amp restricted code)?
//#Expects: Error: test.cpp\(24\).?:.*(\bwchar_t\b).*(is not a supported integral type)?
//#Expects: Error: test.cpp\(25\).?:.*(\bshort\b).*(is not a supported integral type)?
//#Expects: Error: test.cpp\(26\).?:.*(long double is not supported)?

#include <amptest.h>
#include <vector>

using std::vector;
using namespace Concurrency;

#define BLOCK_DIM 16

class c1
{
public:
    wchar_t c; // not allowed here
    short int si;
    long double ud;
    int i;
};

struct FunctObj
{
    FunctObj(array<c1, 2> &fA):mA(fA) {}

    void operator()(tiled_index<BLOCK_DIM, BLOCK_DIM> idx) __GPU_ONLY
    {
        c1 *p1 = &mA[idx];
    }

private:
    array<c1, 2> &mA;
};

runall_result test_main()
{
    srand(2009);
    const int M = 256;

    vector<c1> A(M * M);

    accelerator device = require_device_with_double(Test::Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    extent<2> e(M, M);

    array<c1, 2> fA(e, A.begin(), rv);

    FunctObj cobj(fA);

    parallel_for_each(e.tile<BLOCK_DIM, BLOCK_DIM>(), cobj);

    printf("%s\n", "Failed!");

    return runall_fail;
}

