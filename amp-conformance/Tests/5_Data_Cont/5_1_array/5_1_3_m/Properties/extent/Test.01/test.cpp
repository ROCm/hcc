// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Using extent function, verify if array's extent matches with another extent using the same extents. Check if it is accessible on GPU</summary>

#include "./../../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = 3;
    const extent<_rank> e1(edata);

	{
		array<_type, _rank> src(e1);

		parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU
        {
			src[idx] = static_cast<_type>(src.get_extent().size());
		});

        if (e1.size() != src.get_extent().size())
            return false;

		// Test that the values set in the p_f_e are set to the same size()
		return VerifyAllSameValue(src, static_cast<_type>(e1.size())) == -1;
	}
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

