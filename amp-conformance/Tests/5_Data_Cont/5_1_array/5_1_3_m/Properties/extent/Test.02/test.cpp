// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using extent, verify if array's extent is a const function</summary>

#include "./../../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    extent<_rank> e1(edata);

	{
		array<_type, _rank> src(e1);

        int _1D = 15;
        // verify we are not modifying src extent. - Here we should be operating on temporary extent variable
        src.get_extent()[0] = _1D;
        std::cout << "Modified " << _1D << " extent : " << src.get_extent()[0] << std::endl;
        if (_1D == src.get_extent()[0])
        {
            return false;
        }
	}

	{
		const array<_type, _rank> src(e1);

        int _1D = 15;
        src.get_extent()[0] = _1D;
        std::cout << "Modified " << _1D << " extent : " << src.get_extent()[0] << std::endl;
        if (_1D == src.get_extent()[0])
        {
            return false;
        }
	}

	return true;
}

int main()
{
    int passed =
        test_feature<int, 1>() && test_feature<int, 2>() && test_feature<int, 5>() &&
        test_feature<float, 1>() && test_feature<float, 2>() && test_feature<float, 5>()
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

