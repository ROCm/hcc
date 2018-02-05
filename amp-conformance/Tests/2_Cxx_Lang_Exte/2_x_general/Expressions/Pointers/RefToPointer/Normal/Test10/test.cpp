// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Use reference to pointer as GPU function parameters. Make sure that the value of pointer value can be changed through the reference.</summary>

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
void test_pointer(type *&rptr, type *ptr) __GPU_ONLY
{
    rptr = ptr;
}

template <typename type>
bool test_pointer2(type *&&rrptr, type *ptr) __GPU_ONLY
{
    rrptr = ptr;

    type_comparer<type> comparer;

    type change_value = (type)CHANGED_VALUE; // amp cannot use global const L-value, so set a tmp variable.

    return (comparer.are_equal(*rrptr, change_value));
}

template <typename type>
type * return_ptr(type &value) __GPU_ONLY
{
    type *ptr = &value;

    return ptr;
}

template <typename type>
bool test_global(accelerator_view &alv)
{
    vector<type> v(DATA_SIZE, (type)INIT_VALUE);
    vector<type> v2(DATA_SIZE, (type)CHANGED_VALUE);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);

    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    extent<1> eret(DATA_SIZE_1D);

    array_view<type, RANK> av(e, v);
    array_view<type, RANK> av2(e, v2);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, e.tile<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE>(), [=](tiled_index<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE> idx) __GPU_ONLY
    {
        type *p = &av[idx];
        type *p2 = &av2[idx];

        test_pointer(p, p2);

        type_comparer<type> comparer;

        type change_value = (type)CHANGED_VALUE; // amp cannot use global const L-value, so set a tmp variable.

        if (!comparer.are_equal(*p, change_value))
            av_ret[0] = WRONG_VALUE;

        p = &av[idx];
        p2 = &av2[idx];

        type init_value = (type)INIT_VALUE;
        if (!test_pointer2<type>(return_ptr<type>(init_value), p2))
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

template <typename type>
bool test_shared(accelerator_view &alv)
{
    vector<type> v(DATA_SIZE, (type)INIT_VALUE);
    vector<type> v2(DATA_SIZE, (type)CHANGED_VALUE);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);

    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    extent<1> eret(DATA_SIZE_1D);

    array_view<type, RANK> av(e, v);
    array_view<type, RANK> av2(e, v2);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, e.tile<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE>(), [=](tiled_index<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE> idx) __GPU_ONLY
    {
        tile_static type shared[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
        tile_static type shared2[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

        shared[idx.local[0]][idx.local[1]][idx.local[2]] = av[idx.global];
        shared2[idx.local[0]][idx.local[1]][idx.local[2]] = av2[idx.global];

        idx.barrier.wait();

        type *p = &shared[idx.local[0]][idx.local[1]][idx.local[2]];
        type *p2 = &shared2[idx.local[0]][idx.local[1]][idx.local[2]];

        test_pointer(p, p2);

        type_comparer<type> comparer;

        type change_value = (type)CHANGED_VALUE; // amp cannot use global const L-value, so set a tmp variable.

        if (!comparer.are_equal(*p, change_value))
            av_ret[0] = WRONG_VALUE;

        p = &shared[idx.local[0]][idx.local[1]][idx.local[2]];
        p2 = &shared2[idx.local[0]][idx.local[1]][idx.local[2]];

        type init_value = (type)INIT_VALUE;

        if (!test_pointer2<type>(return_ptr<type>(init_value), p2))
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

template <typename type>
bool test_local(accelerator_view &alv)
{
    vector<type> v(DATA_SIZE, (type)INIT_VALUE);
    vector<type> v2(DATA_SIZE, (type)CHANGED_VALUE);
    vector<int> ret(DATA_SIZE_1D, INIT_VALUE);

    extent<RANK> e(DATA_SIZE_1D, DATA_SIZE_1D, DATA_SIZE_1D);
    extent<1> eret(DATA_SIZE_1D);

    array_view<type, RANK> av(e, v);
    array_view<type, RANK> av2(e, v2);
    array_view<int> av_ret(eret, ret);

    parallel_for_each(alv, e.tile<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE>(), [=](tiled_index<BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE> idx) __GPU_ONLY
    {
        type local[LOCAL_SIZE][LOCAL_SIZE][LOCAL_SIZE];
        type local2[LOCAL_SIZE][LOCAL_SIZE][LOCAL_SIZE];

        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            for (int j = 0; j < LOCAL_SIZE; j++)
            {
                for (int k = 0; k < LOCAL_SIZE; k++)
                {
                    local[i][j][k] = av[idx.global];
                    local2[i][j][k] = av2[idx.global];
                }
            }
        }

        bool failed = false;
        for (int i = 0; i < LOCAL_SIZE; i++)
        {
            for (int j = 0; j < LOCAL_SIZE; j++)
            {
                for (int k = 0; k < LOCAL_SIZE; k++)
                {
                    type *p = &local[i][j][k];
                    type *p2 = &local2[i][j][k];

                    test_pointer(p, p2);

                    type_comparer<type> comparer;

                    type change_value = (type)CHANGED_VALUE; // amp cannot use global const L-value, so set a tmp variable.

                    if (!comparer.are_equal(*p, change_value))
                    {
                        av_ret[0] = WRONG_VALUE;
                        failed = true;
                        break;
                    }

                    p = &local[i][j][k];
                    p2 = &local2[i][j][k];

                    type init_value = (type)INIT_VALUE;

                    if (!test_pointer2<type>(return_ptr<type>(init_value), p2))
                    {
                        av_ret[0] = WRONG_VALUE;
                        failed = true;
                        break;
                    }
                }

                if (failed)
                    break;
            }

            if (failed)
                break;
        }
        if (failed)
            av_ret[0] = WRONG_VALUE;
    });

    av_ret.synchronize();

    return (VerifyAllSameValue<int>(av_ret, INIT_VALUE) == 0 ? false : true);
}

runall_result test_main()
{
    accelerator_view alv = require_device_for<AMP_ELEMENT_TYPE>(device_flags::NOT_SPECIFIED, false).get_default_view();

    runall_result ret;

    ret &= REPORT_RESULT(test_global<AMP_ELEMENT_TYPE>(alv));
    ret &= REPORT_RESULT(test_shared<AMP_ELEMENT_TYPE>(alv));
    ret &= REPORT_RESULT(test_local<AMP_ELEMENT_TYPE>(alv));

    return ret;
}

