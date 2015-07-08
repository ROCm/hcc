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
/// <summary>Check whether the tiled_index is constructed correctly</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

template<int DIM0, int DIM1, int DIM2, int BLOCK0, int BLOCK1, int BLOCK2>
runall_result test(vector<int> &C1, vector<int> &C2, const vector<int> &refC, accelerator_view av)
{
    extent<3> cube(DIM0, DIM1, DIM2);
    array<int, 3> fC1(cube, av);
    array<int, 3> fC2(cube, av);

    parallel_for_each(cube.tile<BLOCK0, BLOCK1, BLOCK2>(), [&] (tiled_index<BLOCK0, BLOCK1, BLOCK2> idxGroup) __GPU
    {
        // c is the flat index calculated from global and group/local indices
        // obtained from the special tiled_index
        index<3> globalIdx = idxGroup;
        index<3> groupIdx = idxGroup.tile;
        index<3> localIdx = idxGroup.local;
        extent<3> groupShape(BLOCK0, BLOCK1, BLOCK2);

        // flat index constructed with group and local index
        int flatIdx = (groupIdx[0] * groupShape[0] + localIdx[0]) * DIM1 * DIM2 +
                      (groupIdx[1] * groupShape[1] + localIdx[1]) * DIM2 +
                      (groupIdx[2] * groupShape[2] + localIdx[2]);

        // flat index constructed with global index
        int flatIdx2 = globalIdx[0] * DIM1 * DIM2 + globalIdx[1] * DIM2 + globalIdx[2];

        fC1[idxGroup] = flatIdx;
        fC2[idxGroup] = flatIdx2;
    });
    C1 = fC1;
    C2 = fC2;

    int verify_size = DIM0 * DIM1 * DIM2;

    runall_result result;
    result &= REPORT_RESULT(Verify(C1.data(), refC.data(), verify_size));
    result &= REPORT_RESULT(Verify(C2.data(), refC.data(), verify_size));

    return result;
}

template<int DIM0, int DIM1, int BLOCK0, int BLOCK1>
runall_result test(vector<int> &C1, vector<int> &C2, const vector<int> &refC, accelerator_view av)
{
    extent<2> matrix(DIM0, DIM1);
    array<int, 2> fC1(matrix, av);
    array<int, 2> fC2(matrix, av);

    parallel_for_each(matrix.tile<BLOCK0, BLOCK1>(), [&] (tiled_index<BLOCK0, BLOCK1> idxGroup) __GPU
    {
        // c is the flat index calculated from global and group/local indices
        // obtained from the special tiled_index
        index<2> globalIdx = idxGroup;
        index<2> groupIdx = idxGroup.tile;
        index<2> localIdx = idxGroup.local;
        extent<2> groupShape(BLOCK0, BLOCK1);

        // flat index constructed with group and local index
        int flatIdx = (groupIdx[0] * groupShape[0] + localIdx[0]) * DIM1 +
                      (groupIdx[1] * groupShape[1] + localIdx[1]);

        // flat index constructed with global index
        int flatIdx2 = globalIdx[0] * DIM1 + globalIdx[1];

        fC1[idxGroup] = flatIdx;
        fC2[idxGroup] = flatIdx2;
    });
    C1 = fC1;
    C2 = fC2;

    int verify_size = DIM0 * DIM1;

    runall_result result;
    result &= REPORT_RESULT(Verify(C1.data(), refC.data(), verify_size));
    result &= REPORT_RESULT(Verify(C2.data(), refC.data(), verify_size));

    return result;
}

template<int DIM0, int BLOCK0>
runall_result test(vector<int> &C1, vector<int> &C2, const vector<int> &refC, accelerator_view av)
{
    extent<1> vec(DIM0);
    array<int, 1> fC1(vec, av);
    array<int, 1> fC2(vec, av);

    parallel_for_each(vec.tile<BLOCK0>(), [&] (tiled_index<BLOCK0> idxGroup) __GPU
    {
        // c is the flat index calculated from global and group/local indices
        // obtained from the special tiled_index
        index<1> globalIdx = idxGroup;
        index<1> groupIdx = idxGroup.tile;
        index<1> localIdx = idxGroup.local;
        extent<1> groupShape(BLOCK0);

        // flat index constructed with group and local index
        int flatIdx = (groupIdx[0] * groupShape[0] + localIdx[0]);

        // flat index constructed with global index
        int flatIdx2 = globalIdx[0];

        fC1[idxGroup] = flatIdx;
        fC2[idxGroup] = flatIdx2;
    });

    C1 = fC1;
    C2 = fC2;

    int verify_size = DIM0;

    runall_result result;
    result &= REPORT_RESULT(Verify(C1.data(), refC.data(), verify_size));
    result &= REPORT_RESULT(Verify(C2.data(), refC.data(), verify_size));

    return result;
}

runall_result test_main()
{
    const int BLOCK0 = 7;
    const int BLOCK1 = 15;
    const int BLOCK2 = 3;
    const int DIM0 = BLOCK0 * 778;
    const int DIM1 = BLOCK1 * 13;
    const int DIM2 = BLOCK2 * 3;

    const unsigned int size = DIM0 * DIM1 * DIM2;

    vector<int> C1(size), C2(size);
    vector<int> refC(size);

    for (unsigned int i = 0; i < size; i++)
    {
        refC[i] = i;
    }

    // test doesn't require double support but require high end cards with high performance
    // to finish compute in less than windows timeout.
    accelerator_view av = require_device(device_flags::D3D11_GPU|device_flags::DOUBLE).get_default_view();

    runall_result result;
    result &= REPORT_RESULT((test<DIM0, BLOCK0>(C1, C2, refC, av)));
    result &= REPORT_RESULT((test<DIM0, DIM1, BLOCK0, BLOCK1>(C1, C2, refC, av)));
    result &= REPORT_RESULT((test<DIM0, DIM1, DIM2, BLOCK0, BLOCK1, BLOCK2>(C1, C2, refC, av)));

    return result;

}

