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
/// <summary>
/// Matrix multiplication using multiple GPUs. Each GPU updates a different section in a tiled fashion using array_views.
/// Arrays are used as underlying data source.
/// </summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

#define BLOCK_SIZE 16
#define BLOCK_SIZEP1 17

void init_matrix(vector<float> & mat, int size)
{
    for (int i = 0; i < size; i++)
    {
        mat[i] = (float)rand()/(float)RAND_MAX;
    }
}

void mxm_cpu_seq(vector<float> & A, vector<float> & B, vector<float> & C,
    int M, int N, int W)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i * N + j ] = 0;
            for (int k = 0; k < W; k++)
            {
                C[i * N + j] += A[i * W + k] * B[k * N + j];
            }
        }
    }
}

index<2> calc_idx(int tileIdx0, int tileIdx1, index<2> localIdx) restrict(amp)
{
    index<2> idx(tileIdx0 * BLOCK_SIZE + localIdx[0],
        tileIdx1 * BLOCK_SIZE + localIdx[1]);
    return idx;
}


void mxm_kernel_tiling(tiled_index<BLOCK_SIZE, BLOCK_SIZE> idxGroup,
    float & c, const array_view<float, 2> &mA, const array_view<float, 2> &mB) restrict(amp)
{
    index<2> tileIdx = idxGroup.tile;
    index<2> localIdx = idxGroup.local;

    float tempC = 0.0f;

    for (int i = 0; i < mA.get_extent()[1]/BLOCK_SIZE; i++)
    {
        tile_static float localA[2*BLOCK_SIZE][BLOCK_SIZEP1];

        index<2> idxA = calc_idx(tileIdx[0], i, localIdx);
        index<2> idxB = calc_idx(i, tileIdx[1], localIdx);

        localA[localIdx[0]][localIdx[1]] = mA[idxA];
        localA[localIdx[0]+BLOCK_SIZE][localIdx[1]] = mB[idxB];

        idxGroup.barrier.wait();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            tempC += localA[localIdx[0]][k] * localA[k+BLOCK_SIZE][localIdx[1]];
        }

        idxGroup.barrier.wait();
    }

    c = tempC;
}


void mxm_tiling(const accelerator_view& av, array_view<float, 2> mC_view, array_view<float, 2> mA_view, array_view<float, 2> mB_view)
{
    parallel_for_each(av, mC_view.get_extent().tile<BLOCK_SIZE, BLOCK_SIZE>(), [=] (tiled_index<BLOCK_SIZE, BLOCK_SIZE> ti) restrict(amp) {
        mxm_kernel_tiling(ti, mC_view[ti], mA_view, mB_view);
    });
}

runall_result test_main()
{
    srand(2009);
    //const int M = 128, N = 256, W = 64;
    const int M = 32, N = 32, W = 32;

    vector<float> A(M * W);
    vector<float> B(W * N);
    vector<float> C(M * N);
    vector<float> refC(M * N);

    init_matrix(A, M * W);
    init_matrix(B, W * N);

	Log(LogType::Info, true) << "Performing matrix multiply on the CPU..." << std::endl;
    mxm_cpu_seq(A, B, refC, M, N, W);
	Log(LogType::Info, true) << "   Done." << std::endl;

	accelerator acc = require_device(device_flags::NOT_SPECIFIED);

	if(acc.get_supports_cpu_shared_memory())
	{
		acc.set_default_cpu_access_type(ACCESS_TYPE);
	}

	accelerator_view av = acc.get_default_view();

    extent<2> eA(M, W), eB(W, N), eC(M, N);
    extent<2> eA_half(M/2, W), eC_half(M/2, N);

    array<float, 2> aA(eA, A.begin(), A.end(), av);
    array<float, 2> aB(eB, B.begin(), B.end(), av);
    array<float, 2> aC(eC, av);

    array_view<float, 2> aA_view1(aA.section(0, 0, M/2, W));
    array_view<float, 2> aA_view2(aA.section(M/2, 0, M/2, W));
    array_view<float, 2> aB_view(aB);
    array_view<float, 2> aC_view1(aC.section(0, 0, M/2, N));
    array_view<float, 2> aC_view2(aC.section(M/2, 0, M/2, N));

	Log(LogType::Info, true) << "Performing matrix multiply on the GPU..." << std::endl;
    Log(LogType::Info, true) << "Starting with View1..." << std::endl;
    mxm_tiling(av, aC_view1, aA_view1, aB);
    Log(LogType::Info, true) << "Starting with View2..." << std::endl;
    mxm_tiling(av, aC_view2, aA_view2, aB);

    C = aC;
	Log(LogType::Info, true) << "   Done." << std::endl;

	return Verify(C, refC);
}

