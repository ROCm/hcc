// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using array* inside parallel_for_each.</summary>
//#Expects: Error: error C3590
//#Expects: Error: error C3581

#include "./../../pointer.h"

void compile_only() {
    size_t numBodies = 1024;

    int *dataSrc = new int[numBodies];
    for (int i = 0; i < numBodies; i++)
        dataSrc[i] = rand();

    extent<1> e(static_cast<int>(numBodies));

    array<int, 1> *pDataA = new array<int, 1>(e, dataSrc);

    parallel_for_each(e, [&](index<1> idx) __GPU_ONLY
    {
        (*pDataA)[idx] = 1;
    });
}

