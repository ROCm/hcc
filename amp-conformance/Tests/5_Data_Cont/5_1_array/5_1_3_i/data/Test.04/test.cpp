// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array's 3-D specialized indexing operators( () ) using GPU index - use set</summary>

#include <set>
#include "./../../index.h"

template<typename _type, int _rank, int _D0, int _D1, int _D2, typename _BeginIterator>
bool test_feature_idx(_BeginIterator _first)
{
    std::vector<_type> src_data;
    std::vector<_type> dst_data;

    {
        array<_type, _rank> src(_D0, _D1, _D2, _first);
        array<_type, _rank> dst(_D0, _D1, _D2);

        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU
        {
            for (int i = 0; i < _D0; i++)
            {
                for (int j = 0; j < _D1; j++)
                {
                    for (int k = 0; k < _D2; k++)
                    {
                        dst(i, j, k) = src(i, j, k);
                        src(i, j, k) = 0;
                    }
                }
            }
        });

        src_data = src;
        dst_data = dst;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || (src_data[i] != 0))
                return false;
        }
    }

    {
        array<_type, _rank> c_src(_D0, _D1, _D2, _first);
        array<_type, _rank> dst1(_D0, _D1, _D2);

        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (int i = 0; i < _D0; i++)
            {
                for (int j = 0; j < _D1; j++)
                {
                    for (int k = 0; k < _D2; k++)
                    {
                        const int& data = c_src(i, j, k);
                        dst1(i, j, k) = data * 2;
                    }
                }
            }
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || ((dst_data[i] != (src_data[i]*2))))
                return false;
        }
    }

    return true;
}

template<typename _type, int _D0, int _D1, int _D2>
bool test_feature()
{
    const int _rank = 3;

    std::multiset<_type> data;
    while(data.size() != _D0*_D1*_D2)
        data.insert((_type)rand());

    return test_feature_idx<_type, _rank, _D0, _D1, _D2>(data.begin());
}

int main()
{
    int passed = test_feature<int, 1, 1, 1>() /*&& test_feature<float, 11, 13, 2>() &&
                    test_feature<double, 2, 31, 31>() && test_feature<unsigned int, 91, 7, 5>() &&
                    test_feature<unsigned, 3, 3, 3>() && test_feature<signed, 111, 2, 3>()*/
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

