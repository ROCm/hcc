//--------------------------------------------------------------------------------------
// File: test.cpp
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
/// <tags>P1</tags>
/// <summary>Tests the parallel_for_each overloads that take an explicit accelerator/accelerator_view argument</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace concurrency::Test;
using std::vector;

const unsigned int MAX_INPUT_VAL = 22;
const unsigned int ARRAY_DIM = 64;
const unsigned int TILE_DIM = 8;
const unsigned int INVALID_TILE_DIM = 1024;

template <int N>
struct dependent_false
{
	static const bool value = false;
};

template <int _Rank>
extent<_Rank> create_extent()
{
    static_assert(dependent_false<_Rank>::value, "Only supported for Ranks > 0 && <=3");
}

template<>
extent<1> create_extent<1>()
{
    return extent<1>(ARRAY_DIM);
}

template<>
extent<2> create_extent<2>()
{
    return extent<2>(ARRAY_DIM, ARRAY_DIM);
}

template<>
extent<3> create_extent<3>()
{
    return extent<3>(ARRAY_DIM, ARRAY_DIM, ARRAY_DIM);
}

// Tests the parallel_for_each overloads that take an accelerator or accelerator_view parameter
// to specify the target, with arrays
template <int _Rank, typename _Target, int _D0 = INVALID_TILE_DIM, int _D1 = 0, int _D2 = 0>
class Test1
{
public:
    static runall_result test(_Target target)
    {
        extent<_Rank> e = create_extent<_Rank>();
        unsigned int size = e.size();

        vector<unsigned int> input(size), expected(size);
        Fill<unsigned int>(input, 0, MAX_INPUT_VAL);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = input[i] + 1;
        }

        array<unsigned int, _Rank> arr(e, input.begin(), target);

        if (_D0 == INVALID_TILE_DIM) {
            parallel_for_each(target, e, [&](index<_Rank> idx) __GPU_ONLY {
                arr[idx] += 1;
            });
        }
        else {
            tiled_extent<_D0, _D1, _D2> tiledGrid(e);
            parallel_for_each(target, tiledGrid, [&](tiled_index<_D0, _D1, _D2> tiled_idx) __GPU_ONLY {
                arr[tiled_idx.global] += 1;
            });
        }

        return VerifyDataOnCpu(arr, expected);
    }
};

// Tests the parallel_for_each overloads that take an accelerator or accelerator_view parameter
// to specify the target, with array_views
template <int _Rank, typename _Target, int _D0 = INVALID_TILE_DIM, int _D1 = 0, int _D2 = 0>
class Test2
{
public:
    static runall_result test(_Target target)
    {
        extent<_Rank> e = create_extent<_Rank>();
        unsigned int size = e.size();

        vector<unsigned int> input(size), expected(size);
        Fill<unsigned int>(input, 0, MAX_INPUT_VAL);
        for(size_t i = 0; i < size; i++)
        {
            expected[i] = input[i] + 1;
        }

        array<unsigned int, _Rank> arr(e, input.begin(), accelerator(accelerator::cpu_accelerator).get_default_view());
        array_view<unsigned int, _Rank> arrView(arr);

        if (_D0 == INVALID_TILE_DIM) {
            parallel_for_each(target, e, [=](index<_Rank> idx) __GPU_ONLY {
                arrView[idx] += 1;
            });
        }
        else {
            //const index<rank>& local, const index<rank> &tile, tile_barrier& barrier
            tiled_extent<_D0, _D1, _D2> tiledGrid(e);
            parallel_for_each(target, tiledGrid, [=](tiled_index<_D0, _D1, _D2> tiled_idx) __GPU_ONLY {
                arrView[tiled_idx.global] += 1;
            });
        }

        return VerifyDataOnCpu(arr, expected);
    }
};

runall_result test_main()
{
    runall_result result;

    result = REPORT_RESULT((Test1<1, accelerator_view>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test1<1, accelerator_view, TILE_DIM>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test1<2, accelerator_view, TILE_DIM, TILE_DIM>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test1<3, accelerator_view, TILE_DIM, TILE_DIM, TILE_DIM>::test(accelerator().create_view())));

    result = REPORT_RESULT((Test2<1, accelerator_view>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test2<1, accelerator_view, TILE_DIM>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test2<2, accelerator_view, TILE_DIM, TILE_DIM>::test(accelerator().create_view())));
    result = REPORT_RESULT((Test2<3, accelerator_view, TILE_DIM, TILE_DIM, TILE_DIM>::test(accelerator().create_view())));

    return result;
}

