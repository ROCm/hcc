// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test the constructor overload accepting an array</summary>

// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;

/* --- All index variables are declared as Index<RANK>(0, 1, 2...) for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */
bool TestOnHost()
{
    int arr[RANK];
    for(int i = 0; i < RANK; i++)
    {
        arr[i] = i;
    }

    index<RANK> idx(arr);    
    return IsIndexSetToSequence<RANK>(idx);
}

/*---------------- Test on Device ---------------- */
/* A returns the components of the index, B returns the rank */
void kernel(array<int, 1>& A, array<int, 1>& B) restrict(amp,cpu)
{
    int arr[RANK];
    for(int i = 0; i < RANK; i++)
    {
        arr[i] = i;
    }

    index<RANK> idx(arr);

    for(int i = 0; i < RANK; i++)
    {
        A(i) = idx[i];        
    }

    B(0) = idx.rank;    
}

bool TestOnDevice()
{
    vector<int> vA(RANK), vB(1);   
    array<int, 1> A((extent<1>(RANK))), B(extent<1>(1));
    extent<1> ex(1);

    parallel_for_each(ex, [&](index<1> idx) restrict(amp,cpu){
        kernel(A, B);
    });
    vA = A;
    vB = B;

    return IsIndexSetToSequence<RANK>(vA, vB[0]);
}

/*--------------------- Main -------------------- */

int main() 
{
    int result = 1;
	 result &= (TestOnHost());
	 result &= (TestOnDevice());
    return !result;
}
