// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify vector conversion operator</summary>

#include "./../../member.h"

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<_type> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = (_type)rand() * (i%5 ? 1 : -1);

    std::vector<_type> src_data;
    std::vector<_type> dst_data;

    {
        array<_type, _rank> src(e1, data.begin());
        array<_type, _rank> dst(e1);

        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst(idx) = src(idx);
        });

        src_data = (vector<_type>)src;
        dst_data = (vector<_type>)dst;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] != src_data[i]) || (dst_data[i] != data[i]))
                return false;
        }
    }

    {
        const array<_type, _rank> src(e1, data.begin());
        array<_type, _rank> dst(e1);

        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst(idx) = src(idx);
        });

        src_data = (vector<_type>)src;
        dst_data = (vector<_type>)dst;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] != src_data[i]) || (dst_data[i] != data[i]))
                return false;
        }
    }

    return true;
}

int main()
{
    int passed = test_feature<int, 5>() && test_feature<float, 5>() &&
                    test_feature<double, 3>() && test_feature<unsigned int, 7>()
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

