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
/// <summary>
/// Matrix multiplication using multiple GPUs. Each GPU updates a different section using array_views.
/// Array is used as underlying data source.
/// </summary>

#include <cstdio>
#include <cstdlib>
#include <amptest.h>
#include <amptest_main.h>
#include <math.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

void InitializeArray(vector<float> &vM, int size)
{
    for(int i=0; i<size; ++i)
    {
        vM[i] = (float)(rand() % RAND_MAX) / (float)RAND_MAX;
    }
}

runall_result test_main()
{
    srand(2010);

    const int M = 8;
    const int N = 64;
    const int W = 8;

    accelerator device1 = require_device(device_flags::NOT_SPECIFIED);
	// Need to get a 2nd device that's different than the first
    accelerator device2 = require_device(device1, device_flags::NOT_SPECIFIED);

	if(device1.get_supports_cpu_shared_memory())
	{
		device1.set_default_cpu_access_type(DEF_ACCESS_TYPE1);
	}

	if(device2.get_supports_cpu_shared_memory())
	{
		device2.set_default_cpu_access_type(DEF_ACCESS_TYPE2);
	}

    accelerator_view av1 = device1.get_default_view();
    accelerator_view av2 = device2.get_default_view();
    accelerator_view av3 = device2.create_view();
    accelerator_view av4 = device2.create_view();

    vector<float> vA(M * N);
    vector<float> vB(N * W);
    vector<float> vC(M * W);
    vector<float> vRef(M * W);

    InitializeArray(vA, M * N);
    InitializeArray(vB, N * W);

    // Compute mxm on CPU
	Log(LogType::Info, true) << "Performing matrix multiply on the CPU..." << std::endl;
    for(int k=0; k<M; ++k)
    {
        for(int j=0; j<W; ++j)
        {
            float result = 0.0f;

            for(int i=0; i<N; ++i)
            {
                int idxA = k * N + i;
                int idxB = i * W + j;

                result += vA[idxA] * vB[idxB];
            }

            vRef[k * W + j] = result;
        }
    }
	Log(LogType::Info, true) << "   Done." << std::endl;

    extent<2> eA(M, N), eB(N, W), eC(M, W);
    extent<2> eA_half(M/2, N), eC_half(M/2, W);

    array<float, 2> mA(eA, vA.begin(), av1);
    array<float, 2> mB(eB, vB.begin(), av2);
    array<float, 2> mC(eC, vC.begin(), av3);

    array_view<float, 2> mA_view1(mA.section(0,0,M/2,N));
    array_view<float, 2> mA_view2(mA.section(M/2,0,M/2,N));
    array_view<float, 2> mB_view(mB);
    array_view<float, 2> mC_view1(mC.section(0,0,M/2,W));
    array_view<float, 2> mC_view2(mC.section(M/2,0,M/2,W));

	Log(LogType::Info, true) << "Performing matrix multiply on the GPU..." << std::endl;
    parallel_for_each(av4, eC_half, [=](index<2> idx) restrict(amp)
        {
            float result = 0.0f;

            for(int i = 0; i < mA_view1.get_extent()[1]; ++i)
            {
                index<2> idxA(idx[0], i);
                index<2> idxB(i, idx[1]);

                result += mA_view1[idxA] * mB_view[idxB];
            }

            mC_view1[idx] = result;
        });


    parallel_for_each(av4, eC_half, [=](index<2> idx) restrict(amp)
        {
            float result = 0.0f;

            for(int i = 0; i < mA_view2.get_extent()[1]; ++i)
            {
                index<2> idxA(idx[0], i);
                index<2> idxB(i, idx[1]);

                result += mA_view2[idxA] * mB_view[idxB];
            }

            mC_view2[idx] = result;
        });

    vC = mC;
	Log(LogType::Info, true) << "   Done." << std::endl;

    // Compare GPU and CPU results
	return Verify(vC, vRef);
}

