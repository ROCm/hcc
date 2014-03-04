// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests for the index assignment operator</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;

/* --- All index variables are declared as Index<RANK>(START, START + 1, START + 2, ...) for ease of verification --- */
const int RANK = 3;
const int START = 11;

/*---------------- Test on Host ------------------ */

int TestOnHost()
{
    int result = 1;

    index<RANK> idx1(START, START + 1, START + 2);
    index<RANK> idx2;
    index<RANK> idx3;

    // check assignment
    idx2 = idx1;
    idx3 = idx1;

    result &= (IsIndexSetToSequence<RANK>(idx2, START));
    result &= (IsIndexSetToSequence<RANK>(idx3, START));

    return result;
}

/*---------------- Test on Device ---------------- */

/* fA, fB will return components of the index and fC returns the rank */
void kernel(array<int, 1>& fA, array<int, 1>& fB, array<int, 1>& fC) restrict(amp,cpu)
{
    index<RANK> idx1(START, START + 1, START + 2);
    index<RANK> idx2;
    index<RANK> idx3;

    // check assignment
    idx2 = idx1;
    idx3 = idx1;

    for(int i = 0; i < RANK; i++)
    {
        fA(i) = idx2[i];
        fB(i) = idx3[i];
    }

    fC(0) = idx2.rank;
    fC(1) = idx3.rank;
}

int TestOnDevice()
{
    int result = 1;


    array<int, 1> A((extent<1>(RANK))), B((extent<1>(RANK))), C((extent<1>(2)));
    extent<1> ex(1);

    parallel_for_each(ex, [&](index<1> idx) restrict(amp,cpu){
        kernel(A, B, C);
    });

    vector<int> vA(RANK), vB(RANK), vC(2);
    vA = A;
    vB = B;
    vC = C;

    result &= (IsIndexSetToSequence<RANK>(vA, vC[0], START));
    result &= (IsIndexSetToSequence<RANK>(vB, vC[1], START));

    return result;
}


/*--------------------- Main -------------------- */

int main() 
{
    int result = 1;
    result &= (TestOnHost());
    result &= (TestOnDevice());
    return !result;
}
