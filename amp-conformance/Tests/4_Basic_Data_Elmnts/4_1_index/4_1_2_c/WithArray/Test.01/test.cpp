// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test the constructor overload accepting an array</summary>


#include "../../../Helpers/IndexHelpers.h"
#include <amptest_main.h>

/* --- All index variables are declared as Index<RANK>(0, 1, 2...) for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */
bool TestOnHost()
{
    Log(LogType::Info, true) << "Testing index contructor with array on host" << std::endl;

    int arr[RANK];
    for(int i = 0; i < RANK; i++)
    {
        arr[i] = i;
    }

    index<RANK> idx(arr);
    return IsIndexSetToSequence<RANK>(idx);
}

/*---------------- Test on Device ---------------- */
/* A returns the components of the index, B returns the rank */
void kernel(array<int, 1>& A, array<int, 1>& B) __GPU
{
    int arr[RANK];
    for(int i = 0; i < RANK; i++)
    {
        arr[i] = i;
    }

    index<RANK> idx(arr);

    for(int i = 0; i < RANK; i++)
    {
        A(i) = idx[i];
    }

    B(0) = idx.rank;
}

bool TestOnDevice()
{
    Log(LogType::Info, true) << "Testing index contructor with array on Device" << std::endl;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    vector<int> vA(RANK), vB(1);
    array<int, 1> A(extent<1>(RANK), av), B(extent<1>(1), av);
    extent<1> ex(1);

    parallel_for_each(ex, [&](index<1> idx) __GPU{
        kernel(A, B);
    });
    vA = A;
    vB = B;

    return IsIndexSetToSequence<RANK>(vA, vB[0]);
}

/*--------------------- Main -------------------- */

runall_result test_main()
{
    runall_result result;
	result &= REPORT_RESULT(TestOnHost());
	result &= REPORT_RESULT(TestOnDevice());
    return result;
}
