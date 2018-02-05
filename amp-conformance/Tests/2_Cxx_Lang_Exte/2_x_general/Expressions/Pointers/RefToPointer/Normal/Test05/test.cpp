// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test pointer operators on lvalue reference to pointers, ++, --, >, >=, <, <=, ==, !=. Use array to test them.</summary>

#include <amptest.h>
#include <amptest_main.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

const int DATA_SIZE_1D = 32;
const int DATA_SIZE = DATA_SIZE_1D * DATA_SIZE_1D * DATA_SIZE_1D;
const int INIT_VALUE = 1;
const int CHANGED_VALUE = 2;
const int WRONG_VALUE = 3;
const int RANK = 3;
const int BLOCK_SIZE = 4;
const int LOCAL_SIZE = 5;

template <typename type>
void init(vector<type> &v)
{
    int cnt = -1;

    std::generate(v.begin(), v.end() , [&cnt]() -> int {
        cnt++;
        return cnt;
    });
}

template <typename type>
bool test_pointer(type *start_address, type *end_address, int data_size) __GPU_ONLY
{
    type_comparer<type> cmp;

    type *p1 = start_address;
    type *&rp1 = p1;

    rp1++;

    if (!cmp.are_equal(*rp1, 1))
    {
        return false;
    }

    ++rp1;

    if (!cmp.are_equal(*rp1, 2))
    {
        return false;
    }

    type *p2 = end_address;
    type *&rp2 = p2;

    if (!cmp.are_equal(rp2[-(data_size -1)], 0))
    {
        return false;
    }

    rp2--;

    if (!cmp.are_equal(*rp2, (type)(data_size - 2)))
    {
        return false;
    }

    --rp2;

    if (!cmp.are_equal(*rp2, (type)(data_size - 3)))
    {
        return false;
    }

    ----rp1; // p1 points to the first data.

    type *p3 = start_address + 1;
    type *&rp3 = p3;

    if ((!(rp1 < rp3)) || (rp1 > rp3) || (rp1 == rp3) || !(rp1 != rp3) || (!(rp1 <= rp3)) || (rp1 >= rp3))
    {
        return false;
    }

    rp1++;

    if (!(rp1 <= rp3) || !(rp1 >= rp3))
    {
        return false;
    }

    return true;
}

template <typename type>
bool test_global(accelerator_view &alv)
{
    vector<type> v(DATA_SIZE);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);

    init(v);

    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    extent<1> eret(DATA_SIZE_1D);

    array_view<type, RANK> av(e, v);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, av.get_extent(), [=](index<RANK>idx) __GPU_ONLY
    {
        if (!test_pointer(&av[0][0][0], &av[DATA_SIZE_1D - 1][DATA_SIZE_1D - 1][DATA_SIZE_1D - 1], DATA_SIZE_1D * DATA_SIZE_1D * DATA_SIZE_1D))
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

template <typename type>
bool test_shared(accelerator_view &alv)
{
    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);
    extent<1> eret(DATA_SIZE_1D);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, e.tile<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE>(), [=](tiled_index<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE> idx) __GPU_ONLY
    {
        tile_static type shared[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

        if ((idx.local[0] == 0) && (idx.local[0] == 0) && (idx.local[0] == 0)) // only the first thread in the tile initialize the data.
        {
            for(int i = 0; i < BLOCK_SIZE; i++)
                for(int j = 0; j < BLOCK_SIZE; j++)
                    for(int k = 0; k < BLOCK_SIZE; k++)
                    {
                        shared[i][j][k] = (type)(i * BLOCK_SIZE * BLOCK_SIZE + j * BLOCK_SIZE + k);
                    }
        }

        idx.barrier.wait();

        if (!test_pointer(&shared[0][0][0], &shared[BLOCK_SIZE - 1][BLOCK_SIZE - 1][BLOCK_SIZE - 1], BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE))
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

template <typename type>
bool test_local(accelerator_view &alv)
{
    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);
    extent<1> eret(DATA_SIZE_1D);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, e.tile<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE>(), [=](tiled_index<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE> idx) __GPU_ONLY
    {
        type local[LOCAL_SIZE][LOCAL_SIZE][LOCAL_SIZE];

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            for (int j = 0; j < LOCAL_SIZE; j++)
            {
                for (int k = 0; k < LOCAL_SIZE; k++)
                {
                    local[i][j][k] = (type)(i * LOCAL_SIZE * LOCAL_SIZE + j * LOCAL_SIZE + k);
                }
            }
        }

        if (!test_pointer(&local[0][0][0], &local[LOCAL_SIZE - 1][LOCAL_SIZE - 1][LOCAL_SIZE - 1], LOCAL_SIZE * LOCAL_SIZE * LOCAL_SIZE))
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

runall_result test_main()
{
    accelerator_view alv = require_device_for<AMP_ELEMENT_TYPE>(device_flags::NOT_SPECIFIED, true).get_default_view();

    runall_result ret;

    ret &= REPORT_RESULT(test_global<AMP_ELEMENT_TYPE>(alv));
    ret &= REPORT_RESULT(test_shared<AMP_ELEMENT_TYPE>(alv));
    ret &= REPORT_RESULT(test_local<AMP_ELEMENT_TYPE>(alv));

    return ret;
}

