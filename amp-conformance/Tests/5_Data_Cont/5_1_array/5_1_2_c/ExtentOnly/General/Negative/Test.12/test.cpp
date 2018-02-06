// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compiler should give error when array is created with extent of different rank - 0, negative integer and rank+1</summary>
//#Expects: Error: error C2664
//#Expects: Error: error C2664
//#Expects: Error: error C2664

#include "./../../../../constructor.h"
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    const int rank = 1;

    int *edata = new int[rank];
    for (int i = 0; i < rank; i++)
        edata[i] = 3;

    {
        extent<rank+1> e1(edata);
        array<_type, rank> src(e1);
    }

    {
        extent<0> e1(edata);
        array<_type, rank> src(e1);
    }

    {
        extent<-1> e1(edata);
        array<_type, rank> src(e1);
    }

    return false;
}

runall_result test_main()
{
    test_feature<int, 3>();

	// We shouldn't compile
    return runall_fail;
}

