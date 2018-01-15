// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using array* inside parallel_for_each captured by value.</summary>


//#Expects: Error: test.cpp\(34\) : error C3596:.*(\bpDataA\b).*(\bConcurrency::array<_Value_type,_Rank> \*)
//#Expects: Error: test.cpp\(34\) : error C3596:.*(\bpDataB\b).*(\bConcurrency::array<_Value_type,_Rank> \*)
//#Expects: Error: test.cpp\(34\) : error C3581:.*(\bmain::<lambda_\w*>)

#include "./../../pointer.h"

int main()
{
    int numBodies = 1024;

    int *dataSrc = new int[numBodies];
    for (int i = 0; i < numBodies; i++)
        dataSrc[i] = rand();

    extent<1> e(numBodies);

    array<int, 1> *pDataA = new array<int, 1>(e, dataSrc);
    array<int, 1> *pDataB = new array<int, 1>(e);

    parallel_for_each(e, [=](index<1> idx) __GPU_ONLY
    {
        (*pDataB)[idx] = (*pDataA)[idx];
    });

    int *dataDst = new int[numBodies];
    copy(*pDataB, dataDst);
    for (int i = 0; i < numBodies; i++)
    {
        if (dataSrc[i] != dataDst[i])
        {
            printf ("src %d dst %d\n", dataSrc[i], dataDst[i]);
            return runall_fail;
        }
    }

    delete[] dataSrc;
    delete[] dataDst;
    delete pDataA;
    delete pDataB;

    printf ("Passed\n");

    return runall_pass;
}


