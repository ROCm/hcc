//--------------------------------------------------------------------------------------
// File: amp.data.h
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License.  You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR
// CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
//
// See the Apache Version 2.0 License for specific language governing permissions
// and limitations under the License.
//
//--------------------------------------------------------------------------------------
//

#pragma once

#include <amp.h>
#include <amptest/logging.h>
#include <amptest/data.h>

namespace Concurrency
{
    namespace Test
    {
		// Generates a random index within the given bounds
        template<int rank>
        index<rank> GetRandomIndex(index<rank>& origin, extent<rank>& ex)
        {
            int subscripts[rank];
            for (int i = 0; i < rank; i++)
            {
                subscripts[i] = origin[i] + (rand() % ex[i]);
            }
            return index<rank>(subscripts);
        }

		// Creates an extent object with range of each dimension between 1 and maxRange.
        template<int _rank>
        extent<_rank> CreateRandomExtent(int max_range)
        {
            int extent_data[_rank];

            for (int i = 0; i < _rank; i++)
			{
				extent_data[i] = 1 + (int)rand() % max_range;
			}

            return extent<_rank>(extent_data);;
        }

        template<typename _type, int _rank>
        array<_type, _rank> CreateStagingArrayAndFillData(const accelerator_view& cpu_av, const accelerator_view& gpu_av, int extent_range)
        {
            if(cpu_av.get_accelerator().get_device_path() != concurrency::accelerator::cpu_accelerator)
			{
				throw amptest_failure("cpu_av is not an accelerator_view on the CPU accelerator.");
			}

			extent<_rank> arr_extent = CreateRandomExtent<_rank>(extent_range);

            std::vector<_type> cont(arr_extent.size());
            Fill<_type>(cont);

            array<_type, _rank> src_arr(arr_extent, cont.begin(), cpu_av, gpu_av);
            Log(LogType::Info, true) << "Created staging array of " << src_arr.get_extent() << std::endl;

            return src_arr;
        }

        template<typename _type, int _rank>
        array<_type, _rank> CreateArrayAndFillData(const accelerator_view& src_av, int extent_range, access_type cpu_access_type = access_type_auto)
        {
            extent<_rank> arr_extent = CreateRandomExtent<_rank>(extent_range);

            std::vector<_type> cont(arr_extent.size());
            Fill<_type>(cont);

            array<_type, _rank> src_arr(arr_extent, cont.begin(), src_av, cpu_access_type);
            Log(LogType::Info, true) << "Created array of " << src_arr.get_extent() << std::endl;

            return src_arr;
        }

        // Creates an array view with non-contiguous data by taking a section on source array as data source.
        template<typename _type, int _rank>
        array_view<_type, _rank> CreateNonContiguousArrayView(array<_type, _rank>& data_src_arr)
        {
            Log(LogType::Info, true) << "Data source array is of " << data_src_arr.get_extent() << std::endl;

            index<_rank> idx;

            for(int i = 0; i < _rank; i++)
            {
                idx[i] = data_src_arr.get_extent()[i] - (1 + (int)rand() % data_src_arr.get_extent()[i]);
            }

            array_view<_type, _rank> non_contig_arr_v  = data_src_arr.section(idx);
            Log(LogType::Info, true) << "Created non-contiguous array view of " << non_contig_arr_v.get_extent() << std::endl;

            return non_contig_arr_v;
        }

		template<typename _type, int _rank>
        array_view<const _type, _rank> CreateNonContiguousArrayViewWithConstType(array<_type, _rank>& data_src_arr)
        {
            Log(LogType::Info, true) << "Data source array is of " << data_src_arr.get_extent();

            index<_rank> idx;

            for(int i = 0; i < _rank; i++)
            {
                idx[i] = data_src_arr.get_extent()[i] - (1 + (int)rand() % data_src_arr.get_extent()[i]);
            }

            array_view<const _type, _rank> non_contig_arr_v  = data_src_arr.section(idx);
            Log(LogType::Info, true) << "Created non-contiguous array view with const type of " << non_contig_arr_v.get_extent() << std::endl;

            return non_contig_arr_v;
        }

    }
}

