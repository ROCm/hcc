// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests for the default index constructor</summary>

#include "../../../Helpers/IndexHelpers.h"
#include <amptest_main.h>

const int RANK = 9;

/*---------------- Test on Host ------------------ */
bool TestOnHost()
{
    Log(LogType::Info, true) << "Testing Index-Default constructor on host" << std::endl;

    index<RANK> idx;
    return IsIndexSetToZero<RANK>(idx);
}

/*---------------- Test on Device ---------------- */

/* A will return components of the index and B returns the rank */
void kernel(array<int, 1>& A, array<int, 1>& B) __GPU
{
    index<RANK> idx;

    for(int i = 0; i < RANK; i++)
    {
        A(i) = idx[i];
    }

    B(0) = idx.rank;
}

bool TestOnDevice()
{
    Log(LogType::Info, true) << "Testing Index-Default constructor on Device" << std::endl;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    array<int, 1> A(extent<1>(RANK), av), B(extent<1>(1), av);
    extent<1> ex(1);

    parallel_for_each(ex, [&](index<1> idx) __GPU{
        kernel(A, B);
    });

    vector<int> vA(RANK), vB(1);
    vA = A;
    vB = B;

    return IsIndexSetToZero<RANK>(vA, vB[0]);
}

runall_result test_main()
{
    runall_result result;
	result &= REPORT_RESULT(TestOnHost());
	result &= REPORT_RESULT(TestOnDevice());
    return result;
}

