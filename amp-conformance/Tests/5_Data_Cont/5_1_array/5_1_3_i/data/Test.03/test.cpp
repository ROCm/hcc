// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array's 2-D specialized indexing operators( () ) using GPU index - use set</summary>

#include <set>
#include "./../../index.h"
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency::Test;

template<typename _type, int _rank, int _D0, int _D1, typename _BeginIterator>
bool test_feature_idx(_BeginIterator _first)
{
    std::vector<_type> src_data;
    std::vector<_type> dst_data;
    _BeginIterator tmpItr;

    std::cout << "Array Dim : " << _D0 << " " << _D1 << std::endl;
    {
        array<_type, _rank> src(_D0, _D1, _first);
        array<_type, _rank> dst(_D0, _D1);

        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (int i = 0; i < _D0; i++)
            {
                for (int j = 0; j < _D1; j++)
                {
                    dst(i, j) = src(i, j);
                    src(i, j) = 0;
                }
            }
        });

        src_data = src;
        dst_data = dst;
        tmpItr = _first;
        for (size_t i = 0; i < dst_data.size(); i++, tmpItr++)
        {
            if ((dst_data[i] == 0) || (src_data[i] != 0))
            {
                std::cout << "Failed src : " << src_data[i] << " dst : " << dst_data[i] << std::endl;
                return false;
            }
            if (dst_data[i] != *tmpItr)
            {
                std::cout << "Failed src : " << src_data[i] << " dst : " << dst_data[i] << std::endl;
                return false;
            }
        }
    }

    {
        array<_type, _rank> c_src(_D0, _D1, _first);
        array<_type, _rank> dst1(_D0, _D1);

        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (int i = 0; i < _D0; i++)
            {
                for (int j = 0; j < _D1; j++)
                {
                    const auto& data = c_src(i, j);
                    dst1(i, j) = data * 2;
                }
            }
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || ((dst_data[i] != (src_data[i]*2))))
            {
                std::cout << "Failed src : " << src_data[i] << " dst : " << dst_data[i] << std::endl;
                return false;
            }
        }
    }
    return true;
}


template<typename _type, int _D0, int _D1>
bool test_feature()
{
    const int _rank = 2;

    std::set<_type> data;
    while(data.size() != _D0*_D1)
        data.insert((_type)rand());

    return test_feature_idx<_type, _rank, _D0, _D1>(data.begin());
}


runall_result test_main()
{
	accelerator::set_default(require_device_for<AMP_TESTVAR_T>(device_flags::NOT_SPECIFIED, false).get_device_path());

    return test_feature<AMP_TESTVAR_T, AMPTESTVAR_D0, AMPTESTVAR_D1>();
}

