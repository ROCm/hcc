// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Tests the copy constructor when it is implicitly called for function parameters</summary>

#include "../../../Helpers/IndexHelpers.h"
#include <amptest_main.h>

/* --- All index variables are declared as Index<RANK> {0, 1, 2 ..} for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */

// Pass index by value to invoke copy constructor
bool func(index<RANK> idx)
{
   return IsIndexSetToSequence<RANK>(idx);
}

bool CopyConstructWithIndexOnHost()
{
    Log(LogType::Info, true) << "Testing copy construct index as function parameter from another index on host" << std::endl;

    index<RANK> idx(0, 1, 2);
    return func(idx);
}

/*---------------- Test on Device ---------------- */

// idx is copy constructed between vector functions
void k1(array<int, 1>& C, array<int, 1>& D, const index<RANK>& idx) __GPU
{
    for(int i = 0; i < RANK;i++)
    {
        C(i) = idx[i];
    }

    D(0) = idx.rank;
}

// idx is copy constructed in the kernel function
void kernel(array<int, 1>& A, array<int, 1>& B, array<int, 1>& C, array<int, 1>& D, const index<RANK>& idx) __GPU
{
    for(int i = 0; i < RANK;i++)
    {
        A(i) = idx[i];
    }

    B(0) = idx.rank;

    k1(C, D, idx);
}

runall_result CopyConstructWithIndexOnDevice()
{
	runall_result result;
    Log(LogType::Info, true) << "Testing copy construct index as parallel_for_each parameter (from another index)" << std::endl;

    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    index<RANK> idxparam(0, 1, 2);

    vector<int> vA(RANK), vB(1), vC(RANK), vD(1);
    array<int, 1> A(extent<1>(RANK), av), B(extent<1>(1), av), C(extent<1>(RANK), av), D(extent<1>(1), av);

    extent<1> ex(1);
    parallel_for_each(ex, [&, idxparam](index<1> idx) __GPU {
        kernel(A, B, C, D, idxparam);
    });

    vA = A;
    vB = B;
    vC = C;
    vD = D;

    result &= REPORT_RESULT(IsIndexSetToSequence<RANK>(vA, vB[0]));
    result &= REPORT_RESULT(IsIndexSetToSequence<RANK>(vC, vD[0]));
    result &= REPORT_RESULT(vB[0] == vD[0]);

    return result;
}


runall_result test_main()
{
	runall_result result;
    result &= REPORT_RESULT(CopyConstructWithIndexOnHost());
    result &= REPORT_RESULT(CopyConstructWithIndexOnDevice());
	return result;
}

