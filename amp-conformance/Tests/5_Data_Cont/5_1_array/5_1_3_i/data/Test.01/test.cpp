// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array's indexing operators( [] and () ) using GPU index - use vector</summary>

#include "./../../index.h"

template<typename _type, int _rank, typename _BeginIterator>
bool test_feature_idx(extent<_rank> _e, _BeginIterator _first)
{
    std::vector<_type> src_data;
    std::vector<_type> dst_data;
    _BeginIterator tmpItr;

    {
        array<_type, _rank> src(_e, _first);
        array<_type, _rank> dst(_e);

        // Used [] operator
        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst[idx] = src[idx];
            src[idx] = 0;
        });

        src_data = src;
        dst_data = dst;
        tmpItr = _first;
        for (size_t i = 0; i < dst_data.size(); i++, tmpItr++)
        {
            if ((dst_data[i] == 0) || (src_data[i] != 0))
                return false;
            if (dst_data[i] != *tmpItr)
                return false;
        }

        // Used () operator
        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            src(idx) = dst(idx);
            dst(idx) = 0;
        });

        src_data = src;
        dst_data = dst;
        tmpItr = _first;
        for (size_t i = 0; i < dst_data.size(); i++, tmpItr++)
        {
            if ((dst_data[i] != 0) || (src_data[i] == 0))
                return false;
            if (src_data[i] != *tmpItr)
                return false;
        }
    }

    {
        array<_type, _rank> c_src(_e, _first);
        array<_type, _rank> dst1(_e);

        // Used [] operator
        parallel_for_each(c_src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            const _type& data = c_src[idx];
            dst1[idx] = data * 2;
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || ((dst_data[i] != (src_data[i]*2))))
                return false;
        }

        // Used () operator
        array<_type, _rank> dst2(_e);
        parallel_for_each(c_src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            const _type& data = c_src(idx);
            dst2(idx) = data * 4;
        });

        src_data = c_src;
        dst_data = dst2;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || ((dst_data[i] != (src_data[i]*4))))
                return false;
        }
    }
    return true;
}

template<int _rank, typename _BeginIterator>
bool test_feature_idx(extent<_rank> _e, _BeginIterator _first)
{
    std::vector<double> src_data;
    std::vector<double> dst_data;
    _BeginIterator tmpItr;

    {
        array<double, _rank> src(_e, _first);
        array<double, _rank> dst(_e);

        // Used [] operator
        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst[idx] = src[idx];
            src[idx] = 0.0;
        });

        src_data = src;
        dst_data = dst;
        tmpItr = _first;
        for (size_t i = 0; i < dst_data.size(); i++, tmpItr++)
        {
            if ((dst_data[i] == 0.0) || (src_data[i] != 0.0))
                return false;
            if (dst_data[i] != *tmpItr)
                return false;
        }

        // Used () operator
        parallel_for_each(src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            src(idx) = dst(idx);
            dst(idx) = 0.0;
        });

        src_data = src;
        dst_data = dst;
        tmpItr = _first;
        for (size_t i = 0; i < dst_data.size(); i++, tmpItr++)
        {
            if ((dst_data[i] != 0.0) || (src_data[i] == 0.0))
                return false;
            if (src_data[i] != *tmpItr)
                return false;
        }
    }

    {
        array<double, _rank> c_src(_e, _first);
        array<double, _rank> dst1(_e);

        // Used [] operator
        parallel_for_each(c_src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst1[idx] = c_src[idx] * 2.0;
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0.0) || ((dst_data[i] != (src_data[i]*2.0))))
                return false;
        }

        // Used () operator
        array<double, _rank> dst2(_e);
        parallel_for_each(c_src.get_extent(), [&] (index<_rank> idx) __GPU_ONLY
        {
            dst2(idx) = c_src(idx) * 4.0;
        });

        src_data = c_src;
        dst_data = dst2;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0.0) || ((dst_data[i] != (src_data[i]*4.0))))
                return false;
        }
    }
    return true;
}


template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<_type> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = (_type)rand();

    return test_feature_idx<_type, _rank>(e1, data.begin());
}

template<int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<double> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = (double)rand();

    return test_feature_idx<_rank>(e1, data.begin());
}

int main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double types.
    // Test is relying on default device, therefore check below is also done on default device.
    accelerator device;
    if (!device.get_supports_limited_double_precision())
    {
        printf("Target device does not support limited double precision\n");
        return 2;
    }

    int passed = test_feature<int, 5>() && test_feature<float, 5>() &&
                    test_feature<5>() && test_feature<unsigned int, 5>() &&
                 test_feature<int, 1>() && test_feature<float, 1>() &&
                    test_feature<1>() && test_feature<unsigned int, 1>() &&
                 test_feature<int, 2>() && test_feature<float, 2>() &&
                    test_feature<2>() && test_feature<unsigned int, 2>() &&
                 test_feature<int, 3>() && test_feature<float, 3>() &&
                    test_feature<3>() && test_feature<unsigned int, 3>()
                    ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

