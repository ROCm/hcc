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
/// <summary>Test that an array_view can be constructed from an AMP array in an amp restricted function</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

const int size = 100;

template<typename T>
void test(index<1> idx, array<T, 1>& data_arr, array<int, 1>& result_arr) restrict(amp)
{
    Concurrency::array_view<T, 1> av(data_arr);

    if(data_arr.get_extent() != av.get_extent()) // Verify extent
    {
        result_arr[idx] = 11;
        return;
    }

    if(data_arr[idx] != av[idx]) // Verify data
    {
        result_arr[idx] = 44;
        return;
    }

    av[idx]++; // update array_view

    if(data_arr[idx] != av[idx]) // check if array is updated
    {
        result_arr[idx] = 55;
        return;
    }

    result_arr[idx] = 100; // All tests passed
}

template<typename T>
int test(accelerator_view acc_view)
{
    vector<int> results_vec(size);
    vector<T> data_vec(size);
    Fill<T>(data_vec.data(), size);

    extent<1> ex(size);


    array<T,1> data_arr(ex, data_vec.begin(), acc_view);
    array<int,1> result_arr(ex, acc_view);

    parallel_for_each(data_arr.get_extent(), [&](index<1> idx) __GPU_ONLY {
        test(idx, data_arr, result_arr);
    });

    results_vec = result_arr;

    // check that all tests passed on the restricted function
    for(int i = 0; i < size;i++)
    {
        if(results_vec[i] != 100)
        {
            printf("Fail: Test %d failed for index %d\n", results_vec[i], i);
            return runall_fail;
        }
    }

    printf("Pass!\n");

    return runall_pass;
}

int main()
{

   accelerator device;
   if (!get_device(Device::ALL_DEVICES, device))
   {
       return runall_skip;
   }
   accelerator_view acc_view = device.get_default_view();

   int result = test<int>(acc_view);

   if(result != runall_pass) return result;

   result = test<float>(acc_view);

   if(result != runall_pass) return result;

   result = test<long>(acc_view);

   if(result != runall_pass) return result;
}

