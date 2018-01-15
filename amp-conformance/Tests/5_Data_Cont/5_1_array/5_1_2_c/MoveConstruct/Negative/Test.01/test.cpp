// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compiler should give error when source of Move constructor is not of same rank of dst</summary>
//#Expects: Error: test.cpp\(36\) : error C2440
//#Expects: Error: test.cpp\(48\) : error C2440

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type>
_type foo(_type src)
{
	return src;
}

template<typename _type, int _rank>
bool test_feature()
{
    const int rank = _rank;

    {
        int *edata1 = new int[rank+1];
        for (int i = 0; i < rank+1; i++)
            edata1[i] = 3;
        extent<rank+1> e1(edata1);

        // src is of different rank
        array<_type, rank+1> src(e1);

        array<_type, rank> dst = std::move(src);
    }

    {
        int *edata = new int[rank];
        for (int i = 0; i < rank; i++)
            edata[i] = 3;
        extent<rank+1> e1(edata);

        // src is of different type
        array<_type, rank+1> src(e1);

        array<_type, rank> dst = foo<array<_type, rank+1>>(src);
    }

    return false;
}

runall_result test_main()
{
    test_feature<int, 1>();

	// We shouldn't compile
    return runall_fail;
}

