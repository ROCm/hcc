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
/// <summary>Test that parallel_for_each is executed correctly for boundary values of tile extent dimensions for 1D, 2D and 3D.</summary>
#include <amptest.h>
#include <amptest_main.h>
using std::vector;
using namespace concurrency;
using namespace concurrency::Test;

template <int Z, int Y, int X, int size>
bool test(const accelerator_view &av)
{
    vector<int> A(size);
    vector<int> B(size);
    vector<int> C(size);
    vector<int> refC(size);

    Test::Fill(A);
    Test::Fill(B);
    for(int i=0; i<size; ++i)
    {
        refC[i] = A[i] + B[i];
    }

    static const int reminder = (size/Z)/Y;
    static_assert(reminder % X == 0, "Non evenly dividable size by dims");
    extent<3> e(Z, Y, reminder);
    array<int, 3> fA(e, av);
    array<int, 3> fB(e, av);
    array<int, 3> fC(e, av);

    copy(A.begin(), A.end(), fA);
    copy(B.begin(), B.end(), fB);
    copy(C.begin(), C.end(), fC);

    parallel_for_each(e.tile<Z, Y, X>(), [&] (tiled_index<Z, Y, X> ti) __GPU
    {
        fC[ti] = fB[ti] + fA[ti];
    });

    C = fC;

    return Test::Verify(C, refC);
}

template <int Y, int X, int size>
bool test(const accelerator_view &av)
{
    vector<int> A(size);
    vector<int> B(size);
    vector<int> C(size);
    vector<int> refC(size);

    Test::Fill(A);
    Test::Fill(B);
    for(int i=0; i<size; ++i)
    {
        refC[i] = A[i] + B[i];
    }

    static const int reminder = size/Y;
    static_assert(reminder % X == 0, "Non evenly dividable size by dims");
    extent<2> e(Y, reminder);
    array<int, 2> fA(e, av);
    array<int, 2> fB(e, av);
    array<int, 2> fC(e, av);

    copy(A.begin(), A.end(), fA);
    copy(B.begin(), B.end(), fB);
    copy(C.begin(), C.end(), fC);

    parallel_for_each(e.tile<Y, X>(), [&] (tiled_index<Y, X> ti) __GPU
    {
        int b = fB[ti];
        fC[ti] = fA[ti] + b;
    });

    C = fC;

    return Test::Verify(C, refC);
}

template <int X, int size>
bool test(const accelerator_view &av)
{
    vector<int> A(size);
    vector<int> B(size);
    vector<int> C(size);
    vector<int> refC(size);

    Test::Fill(A);
    Test::Fill(B);
    for(int i=0; i<size; ++i)
    {
        refC[i] = A[i] + B[i];
    }

    static_assert(size % X == 0, "Non evenly dividable size by dims");
    extent<1> e(size);
    array<int, 1> fA(e, av);
    array<int, 1> fB(e, av);
    array<int, 1> fC(e, av);

    copy(A.begin(), A.end(), fA);
    copy(B.begin(), B.end(), fB);
    copy(C.begin(), C.end(), fC);

    parallel_for_each(e.tile<X>(), [&] (tiled_index<X> ti) __GPU
    {
        int b = fB[ti];
        fC[ti] = fA[ti] + b;
    });

    C = fC;

    return Test::Verify(C, refC);
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    runall_result result;

    //X in 3D
	result &= REPORT_RESULT((test<1, 1, 1024, 1024>(av)));
	result &= REPORT_RESULT((test<1, 1, 1024, 2048>(av)));

    //Y in 3D
	result &= REPORT_RESULT((test<1, 1024, 1, 1024>(av)));

    //Z in 3D
	result &= REPORT_RESULT((test<64, 1, 1, 64>(av)));

    //3D mixes
	result &= REPORT_RESULT((test<64, 4, 4, 1024>(av))); // 4 * 4 * 64 = 1024
	result &= REPORT_RESULT((test<64, 4, 4, 2048>(av)));

    //X in 2D
	result &= REPORT_RESULT((test<1, 1024, 1024>(av)));
	result &= REPORT_RESULT((test<1, 1024, 2048>(av)));

    //Y in 2D
	result &= REPORT_RESULT((test<1024, 1, 1024>(av)));

	//2D mixes
	result &= REPORT_RESULT((test<128, 8, 2048>(av))); // 128 * 8 = 1024

	//1D
	result &= REPORT_RESULT((test<1024, 1024>(av)));
	result &= REPORT_RESULT((test<1024, 2048>(av)));

    return result;
}
