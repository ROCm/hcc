// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify array's 1-D specialized indexing operators( [] and () ) using GPU index - use deque</summary>

#include <deque>
#include "./../../index.h"

template<typename _type, int _rank, int _D0, typename _BeginIterator>
bool test_feature_idx(_BeginIterator _first)
{
    std::vector<_type> src_data;
    std::vector<_type> dst_data;
    _BeginIterator tmpItr;

    {
        array<_type, _rank> src(_D0, _first);
        array<_type, _rank> dst(_D0);

        // Used [] operator
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                dst[i] = src[i];
                src[i] = 0;
            }
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
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                src(i) = dst(i);
                dst(i) = 0;
            }
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
        array<_type, _rank> c_src(_D0, _first);
        array<_type, _rank> dst1(_D0);

        // Used [] operator
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
             for (unsigned int i = 0; i < c_src.get_extent().size(); i++)
             {
                const _type& data = c_src[i];
                dst1[i] = data * 2;
             }
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0) || ((dst_data[i] != (src_data[i]*2))))
                return false;
        }

        // Used () operator
        array<_type, _rank> dst2(_D0);
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < c_src.get_extent().size(); i++)
            {
                const _type& data = c_src(i);
                dst2(i) = data * 4;
            }
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

template<int _rank, int _D0, typename _BeginIterator>
bool test_feature_idx(_BeginIterator _first)
{
    std::vector<double> src_data;
    std::vector<double> dst_data;
    _BeginIterator tmpItr;

    {
        array<double, _rank> src(_D0, _first);
        array<double, _rank> dst(_D0);

        // Used [] operator
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                dst[i] = src[i];
                src[i] = 0.0;
            }
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
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                src(i) = dst(i);
                dst(i) = 0.0;
            }
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
        array<double, _rank> c_src(_D0, _first);
        array<double, _rank> dst1(_D0);

        // Used [] operator
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
             for (unsigned int i = 0; i < c_src.get_extent().size(); i++)
             {
                dst1[i] = c_src[i] * 2.0;
             }
        });

        src_data = c_src;
        dst_data = dst1;
        for (size_t i = 0; i < dst_data.size(); i++)
        {
            if ((dst_data[i] == 0.0) || ((dst_data[i] != (src_data[i]*2.0))))
                return false;
        }

        // Used () operator
        array<double, _rank> dst2(_D0);
        parallel_for_each(extent<1> (1), [&] (index<1>) __GPU_ONLY
        {
            for (unsigned int i = 0; i < c_src.get_extent().size(); i++)
            {
                dst2(i) = c_src(i) * 4.0;
            }
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

template<typename _type, int _D0>
bool test_feature()
{
    const int _rank = 1;

    std::deque<_type> data(_D0);
    for (int i = 0; i < _D0; i++)
        data[i] = (_type)rand();

    return test_feature_idx<_type, _rank, _D0>(data.begin());
}

template<int _D0>
bool test_feature()
{
    const int _rank = 1;

    std::deque<double> data(_D0);
    for (int i = 0; i < _D0; i++)
        data[i] = (double)rand();

    return test_feature_idx<_rank, _D0>(data.begin());
}

int main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double precision.
    // Please note that test is relying on default device, therefore check below is also done on default device.
    accelerator device;
    if (!device.get_supports_limited_double_precision())
    {
        printf("Default device does not support limited double precision\n");
        return 2;
    }

    int passed = test_feature<int, 1>() && test_feature<float, 5>() &&
                    test_feature<31>() && test_feature<unsigned int, 91>()
            ? runall_pass : runall_fail;

    printf("%s\n", (passed == runall_pass) ? "Passed!" : "Failed!");

    return passed;
}

