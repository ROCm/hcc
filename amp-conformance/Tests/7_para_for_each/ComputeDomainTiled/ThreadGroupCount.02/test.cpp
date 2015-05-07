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
/// <summary>ThreadGroupCount tests for grouped 3D parallel_for_each</summary>

#include <iostream>
#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using std::vector;
using namespace Concurrency;

// This template allows us to control dimension and compute domain size at each position in 3D grouped parallel_for_each.
template<int D0, int D1, int D2, int C0, int C1, int C2>
bool test3(const accelerator_view &av)
{
    printf("test3: %d %d %d %d %d %d : ", D0, D1, D2, C0, C1, C2);

    const int size = 10;
    const int totalsize = size * size * size;
    const int expectedValue = 7777;

    vector<int> C(totalsize);
    vector<int> refC(totalsize);

    for(int i=0; i<totalsize; ++i)
    {
        C[i] = 0;
        refC[i] = 0;

        // in some scenarios our compute domain might be smaller than array
        if (i < C0 * C1 * C2)
        {
            refC[i] = expectedValue;
        }
    }

    extent<3> e(size, size, size);
    array<int, 3> fC(e, C.begin(), C.end(), av);

    parallel_for_each(extent<3>(C0, C1, C2).tile<D0, D1, D2>(), [&,totalsize,size,expectedValue](tiled_index<D0, D1, D2> ti) __GPU
    {
       // compute flat index
       index<3> global = ti;
       int flatIdx = global[0] * ti.tile_dim1 * ti.tile_dim2 + global[1] * ti.tile_dim2 + global[2];

       // prevent indexing out of bounds if compute domain is greater than array size
       if(flatIdx < totalsize)
       {
            // compute array index based on flat index
            // notice that it would not be the same as using idx.global
            int i0 = (flatIdx/size)/size;
            flatIdx -= i0 * size * size;
            int i1 = flatIdx/size;
            flatIdx -= i1 * size;
            int i2 = flatIdx;
            index<3> i(i0, i1, i2);

            fC[i] = expectedValue;
       }
    });

    C = fC;

    bool passed = Test::Verify(C, refC);
    printf(passed ? "passed\n" : "failed\n");

    return passed;
}

runall_result test_main()
{
    accelerator_view av = require_device(Test::device_flags::D3D11_GPU|Test::device_flags::D3D11_WARP).get_default_view();

    static const int maxThreadGroupCount = 65535;

    bool passed = true;

    try
    {
        // max groupcount on selected dims
        passed = test3<1, 1, 1, maxThreadGroupCount, 1, 1>(av) ? passed : false;
        passed = test3<1, 1, 1, 1, maxThreadGroupCount, 1>(av) ? passed : false;
        passed = test3<1, 1, 1, 1, 1, maxThreadGroupCount>(av) ? passed : false;

        // special case - smallest possible thread group count and thread group size
        passed = test3<1, 1, 1, 1, 1, 1>(av) ? passed : false;

        // max threadgroupcount on D0 and max threads in Z dim
        passed = test3<64, 1, 1, 64*maxThreadGroupCount, 1, 1>(av) ? passed : false;
        // max threadgroupcount on D0, and max threads (1024 = 64 * 16)
        passed = test3<64, 16, 1, 64*maxThreadGroupCount, 16, 1>(av) ? passed : false;
        // max threadgroupcount on D0, and max threads (1024 = 64 * 16)
        passed = test3<64, 1, 16, 64*maxThreadGroupCount, 1, 16>(av) ? passed : false;
        // max threadgroupcount on D0, and max threads (1024 = 64 * 4 * 4)
        passed = test3<64, 4, 4, 64*maxThreadGroupCount, 4, 4>(av) ? passed : false;

        // max threadgroupcount on D1 and max threads
        passed = test3<1, 1024, 1, 1, 1024*maxThreadGroupCount, 1>(av) ? passed : false;
        // max threadgroupcount on D0, D1, D2 and max threads

        // max threadgroupcount on D3 and max threads
        passed = test3<1, 1, 1024, 1, 1, 1024*maxThreadGroupCount>(av) ? passed : false;
    }
    catch(const std::exception &e)
    {
        printf("Caught exception: %s\n", e.what());
        passed = false;
    }

    printf("Test: %s\n", passed? "Passed!" : "Failed!");

    return runall_result(passed);
}

