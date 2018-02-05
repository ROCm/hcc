// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compiler should give error when source of Move constructor is not of same type of dst</summary>
//#Expects: Error: test.cpp\(36\) : error C2440
//#Expects: Error: test.cpp\(54\) : error C2440

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type>
_type foo(_type src)
{
	return src;
}

template<typename _type, int _rank, typename _othertype>
bool test_feature()
{
    const int rank = _rank;

    {
        int *edata = new int[rank];
        for (int i = 0; i < rank; i++)
            edata[i] = 3;
        extent<rank> e1(edata);

        // src is of different type
        array<_othertype, rank> src(e1);

        array<_type, rank> dst = std::move(src);

        if (dst.get_extent() != e1)
        {
            return false;
        }
    }


    {
        int *edata = new int[rank];
        for (int i = 0; i < rank; i++)
            edata[i] = 3;
        extent<rank> e1(edata);

        // src is of different type
        array<_othertype, rank> src(e1);

        array<_type, rank> dst = foo<array<_othertype, rank>>(src);

        if (dst.get_extent() != e1)
        {
            return false;
        }
    }

    return false;
}

runall_result test_main()
{
    test_feature<int, 1, float>();

	// We shouldn't compile
    return runall_fail;
}

