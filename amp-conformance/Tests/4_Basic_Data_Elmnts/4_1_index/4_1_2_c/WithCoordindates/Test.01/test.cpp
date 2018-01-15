// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests that checks the index constructor taking individual index components</summary>

#include "../../../Helpers/IndexHelpers.h"
#include <amptest_main.h>

/* --- All index variables are declared as Index<N>(0, 1, 2, ...) for ease of verification --- */

/*---------------- Test on Host ------------------ */
runall_result TestOnHost()
{
	runall_result result;
    Log(LogType::Info, true) << "Testing constructor that takes individual co-ordinates on host" << std::endl;

    index<1> idx1(0);
    index<2> idx2(0, 1);
    index<3> idx3(0, 1, 2);

    result &= REPORT_RESULT(IsIndexSetToSequence<1>(idx1));
    result &= REPORT_RESULT(IsIndexSetToSequence<2>(idx2));
    result &= REPORT_RESULT(IsIndexSetToSequence<3>(idx3));
    return result;
}

/*---------------- Test on Device ---------------- */

/* A, B, C return the components of each index. D returns all the rank values */
void kernel(array<int, 1>& A, array<int, 1>& B, array<int, 1>& C, array<int, 1>& D) __GPU
{
    index<1> idx1(0);
    index<2> idx2(0, 1);
    index<3> idx3(0, 1, 2);

    A(0) = idx1[0];
    D(0) = idx1.rank;

    for(int i = 0; i < 2; i++)
    {
        B(i) = idx2[i];
    }
    D(1) = idx2.rank;

    for(int i = 0; i < 3; i++)
    {
        C(i) = idx3[i];
    }
    D(2) = idx3.rank;
}

runall_result TestOnDevice()
{
	runall_result result;
    Log(LogType::Info, true) << "Testing constructor that takes individual co-ordinates on Device" << std::endl;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    /* vA, vB, vC, vD hold the components of each index. vE, holds all the rank values */
    vector<int> vA(1), vB(2), vC(3), vD(3);
    array<int, 1> A(extent<1>(1), av), B(extent<1>(2), av), C(extent<1>(3), av), D(extent<1>(3), av);

    extent<1> ex(1);
    parallel_for_each(ex, [&](index<1> idx) __GPU {
        kernel(A, B, C, D);
    });

    vA = A;
    vB = B;
    vC = C;
    vD = D;

	result &= REPORT_RESULT(IsIndexSetToSequence<1>(vA, vD[0]));
    result &= REPORT_RESULT(IsIndexSetToSequence<2>(vB, vD[1]));
    result &= REPORT_RESULT(IsIndexSetToSequence<3>(vC, vD[2]));

    return result;
}

/*--------------------- Main -------------------- */
runall_result test_main()
{
    runall_result result;
	result &= REPORT_RESULT(TestOnHost());
	result &= REPORT_RESULT(TestOnDevice());
    return result;
}

