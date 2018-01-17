// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using array's copy assignment - verify compiler error message when assigning to const array</summary>
//#Expects: Error: error C2678
//#Expects: Error: error C2678

#include "./../../../constructor.h"
#include <amptest_main.h>

template<typename _type>
_type foo(const _type src)
{
	return src;
}

template<typename _type, int _rank>
bool test_feature()
{
    const int rank = _rank;

    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    extent<rank> e1(edata);

    {
		std::vector<_type> data;
		for (decltype(e1.size()) i = 0; i != e1.size(); ++i)
			data.push_back((_type)rand());
        array<_type, rank> src(e1, data);
        const array<_type, rank> dst(e1);

		// copy assignment
		dst = src;
    }

    {
		std::vector<_type> data;
		for (decltype(e1.size()) i = 0; i != e1.size(); ++i)
			data.push_back((_type)rand());
        const array<_type, rank> src(e1, data);
        const array<_type, rank> dst(e1);

		// copy assignment
		dst = src;
    }

    return false;
}

runall_result test_main()
{
    test_feature<int, 1>();

	// We shouldn't compile
    return runall_fail;
}

