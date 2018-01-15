// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Verify function data() on device</summary>

#include <cstdint>
#include <iterator>
#include "./../../member.h"
#include <amptest.h>
#include <amptest_main.h>

template<typename _type, int _rank>
bool test_feature()
{
    int edata[_rank];
    for (int i = 0; i < _rank; i++)
        edata[i] = i+1;
    extent<_rank> e1(edata);

    std::vector<_type> data(e1.size());
    for (unsigned int i = 0; i < e1.size(); i++)
        data[i] = static_cast<_type>(i+1);

    {
        array<_type, _rank> src(e1, data.begin(), data.end());
        array<_type, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            _type* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                if (dst_data[i] != static_cast<_type>(i+1))
                    res[idx] = 1;
                else
                    res[idx] = 0;
            }
        });

        std::vector<_type> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1)
                return false;
        }
    }

    {
        array<_type, _rank> src(e1, data.begin(), data.end());
        array<_type, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            const _type* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                if (dst_data[i] != static_cast<_type>(i+1))
                    res[idx] = 1;
                else
                    res[idx] = 0;
            }
        });

        std::vector<_type> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1)
                return false;
        }
    }

    {
        const array<_type, _rank> src(e1, data.begin(), data.end());
        array<_type, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            const _type* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                if (dst_data[i] != static_cast<_type>(i+1))
                    res[idx] = 1;
                else
                    res[idx] = 0;
            }
        });

        std::vector<_type> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1)
                return false;
        }
    }

    return true;
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
        data[i] = i+1;

    {
        array<double, _rank> src(e1, data.begin(), data.end());
        array<double, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            double* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                float x = static_cast<float>(i + 1);
                if (dst_data[i] != x)
                    res[idx] = 1.0;
                else
                    res[idx] = 0.0;
            }
        });

        std::vector<double> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1.0)
                return false;
        }
    }

    {
        array<double, _rank> src(e1, data.begin(), data.end());
        array<double, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            const double* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                float x = static_cast<float>(i + 1);
                if (dst_data[i] != x)
                    res[idx] = 1.0;
                else
                    res[idx] = 0.0;
            }
        });

        std::vector<double> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1.0)
                return false;
        }
    }

    {
        const array<double, _rank> src(e1, data.begin(), data.end());
        array<double, _rank> res(e1);

        parallel_for_each(src.get_extent(), [&](index<_rank> idx) __GPU_ONLY
        {
            const double* dst_data = src.data();

            for(unsigned int i = 0; i < src.get_extent().size(); i++)
            {
                float x = static_cast<float>(i + 1);
                if (dst_data[i] != x)
                    res[idx] = 1.0;
                else
                    res[idx] = 0.0;
            }
        });

        std::vector<double> res_data = res;
        for (size_t i = 0; i < res_data.size(); i++)
        {
            if (res_data[i] == 1.0)
                return false;
        }
    }

    return true;
}

runall_result test_main()
{
    // Test is using doubles therefore we have to make sure that it is not executed
    // on devices that does not support double types.
    // Test is relying on default device, therefore check below is also done on default device.
    accelerator device;
    if (!device.get_supports_limited_double_precision())
    {
        printf("Default device does not support limited double precision\n");
        return runall_skip;
    }

	runall_result res;
	
	res &= REPORT_RESULT((test_feature<int, 3>()));
	res &= REPORT_RESULT((test_feature<float, 5>()));
	res &= REPORT_RESULT((test_feature<3>()));
	res &= REPORT_RESULT((test_feature<int32_t, 5>()));
	
    return res;
}

