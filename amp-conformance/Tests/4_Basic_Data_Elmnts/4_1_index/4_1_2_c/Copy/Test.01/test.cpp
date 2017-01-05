// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests for the index copy constructor</summary>

#include "../../../Helpers/IndexHelpers.h"
#include <amptest_main.h>

/* --- All index variables are declared as Index<RANK> {0, 1, 2..} for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */
bool CopyConstructWithIndexOnHost()
{
    Log(LogType::Info, true)<< "Testing copy construct index with another index on host" << std::endl;

    index<RANK> idx1(0, 1, 2);
    index<RANK> idx2(idx1);   // copy construct

    return IsIndexSetToSequence<RANK>(idx2);
}

/*---------------- Test on Device ---------------- */
/* A returns the components of the index, B returns the Rank */
void kernelIndex(array<int, 1>& A, array<int, 1>& B) __GPU
{
    index<RANK> index1(0, 1, 2);
    index<RANK> index2(index1);   // copy construct

    for(int i = 0; i < RANK;i++)
    {
        A(i) = index2[i];
    }

    B(0) = index2.rank;
}

bool CopyConstructWithIndexOnDevice()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    Log(LogType::Info, true)<< "Testing copy construct index with an index on device" << std::endl;

    vector<int> resultsA(RANK), resultsB(1);
    array<int, 1> A(extent<1>(RANK), av), B(extent<1>(1), av);

    Concurrency::extent<1> ex(1);
    parallel_for_each(ex, [&](index<1> idx) __GPU{
        kernelIndex(A, B);
    });

    resultsA = A;
    resultsB = B;

    return IsIndexSetToSequence<RANK>(resultsA, resultsB[0]);
}


/*--------------------- Main -------------------- */
runall_result test_main()
{
    runall_result result;

    // Test on host
    result &= REPORT_RESULT(CopyConstructWithIndexOnHost());

    // Test on device
    result &= REPORT_RESULT(CopyConstructWithIndexOnDevice());
    return result;
}

