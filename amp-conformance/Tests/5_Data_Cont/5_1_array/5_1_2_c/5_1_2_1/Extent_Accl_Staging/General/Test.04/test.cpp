// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Naive matrix mult using staging arrays</summary>

#include <amptest.h>
#include <stdio.h>

using namespace Concurrency;
using namespace Concurrency::Test;

const unsigned int MAX_INPUT_VAL = 22;

bool Test1(const accelerator& acc)
{
    const unsigned int ARRAY_DIM = 256;

    extent<2> e(ARRAY_DIM, ARRAY_DIM);
    array<unsigned int, 2> sA(e, accelerator(accelerator::cpu_accelerator).get_default_view(), acc.get_default_view());
    array<unsigned int, 2> sB(e, accelerator(accelerator::cpu_accelerator).get_default_view(), acc.get_default_view());

    unsigned int *pA = sA.data();
    unsigned int *pB = sB.data();
    for (size_t i = 0; i < ARRAY_DIM * ARRAY_DIM; ++i) {
        pA[i] = rand() % MAX_INPUT_VAL;
        pB[i] = rand() % MAX_INPUT_VAL;
    }


    array<unsigned int, 2> mA(e, acc.get_default_view());
    copy(sA, mA);

    array<unsigned int, 2> mB(e, acc.get_default_view());
    copy(sB, mB);

    array<unsigned int, 2> mC(e, acc.get_default_view());

    parallel_for_each(mC.get_extent(), [&](index<2> idx) __GPU
    {
        unsigned int result = 0;

        for(int i = 0; i < mA.get_extent()[1]; ++i)
        {
            result += mA(idx[0], i) * mB(i, idx[1]);
        }

        mC[idx] = result;
    });

    array<unsigned int, 2> sC(e, accelerator(accelerator::cpu_accelerator).get_default_view(), acc.get_default_view());
    copy(mC, sC);

    // Compute on CPU
    pA = sA.data();
    array<unsigned int, 2> ref(e, accelerator(accelerator::cpu_accelerator).get_default_view());
    for (int i = 0; i < ARRAY_DIM; ++i) {
        for (int j = 0; j < ARRAY_DIM; ++j) {
            unsigned int result = 0;
            for (int k = 0; k < ARRAY_DIM; ++k) {
                result += pA[(i * ARRAY_DIM) + k] * sB(k, j);
            }

            ref(i, j) = result;
        }
    }

    // Verify the result
    bool passed = true;
    unsigned int *pC = sC.data();
    for (int i = 0; i < ARRAY_DIM; ++i) {
        for (int j = 0; j < ARRAY_DIM; ++j) {
            if (pC[(i * ARRAY_DIM) + j] != ref(i, j)) {
                printf("Incorrect result at (%d, %d): Expected %d, got %d!\n", i, j, ref(i, j), pC[(i * ARRAY_DIM) + j]);
                passed = false;
            }
        }
    }

    return passed;
}

int main()
{
    bool passed = true;

    accelerator acc;
    if (!get_device(Device::ALL_DEVICES, acc))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }

    passed = Test1(acc) ? passed : false;

    printf("%s!\n", passed ? "Passed" : "Failed");

    return passed ? 0 : 1;
}

