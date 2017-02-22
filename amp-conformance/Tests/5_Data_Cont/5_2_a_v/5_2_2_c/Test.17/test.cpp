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
/// <summary>Test that we can create an array_view with data of struct type (PODs only)</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

struct struct_a
{
    int a_a;
    float a_b;
    unsigned int a_c;
};

struct struct_b : struct_a
{
    int b_a;
    float b_b;
    unsigned int b_c;
};

int main()
{
    const int size = 100;

    vector<struct_b> vec(size);
    for(int i = 0; i < size; i++)
    {
        vec[i].a_a = i + 1;
        vec[i].a_b = static_cast<float>(i + 2);
        vec[i].a_c = i + 3;
        vec[i].b_a = i + 4;
        vec[i].b_b = static_cast<float>(i + 5);
        vec[i].b_c = i + 6;
    }

    extent<1> ex(size);
    array<struct_b, 1> arr(ex, vec.begin());
    array_view<struct_b, 1> av(arr);

    if(arr.get_extent() != av.get_extent())    // verify extent
    {
        printf("array and array_view extents do not match. FAIL!\n");
        return runall_fail;
    }

    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return runall_skip;
    }
    accelerator_view acc_view = device.get_default_view();

    // use in parallel_for_each
    parallel_for_each(acc_view, av.get_extent(), [=] (index<1> idx) __GPU
    {
        av[idx].a_a++;  av[idx].a_b++; av[idx].a_c++;
        av[idx].b_a++;  av[idx].b_b++; av[idx].b_c++;
    });

    // arr should automatically be updated at this point
    vec = arr;

    // verify data
    for(int i = 0; i < size; i++)
    {
        if(vec[i].a_a != (i+2) || av[i].a_a != (i+2))
        {
            printf("a_a data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%d] Actual av : [%d]\n", i+2, vec[i].a_a, av[i].a_a);
            return runall_fail;
        }

        if(!AreAlmostEqual(vec[i].a_b, static_cast<float>(i+3)) || !AreAlmostEqual(av[i].a_b, static_cast<float>(i+3)))
        {
            printf("a_b data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%f] Actual av : [%f]\n", i+3, vec[i].a_b, av[i].a_b);
            return runall_fail;
        }

        if(vec[i].a_c != static_cast<unsigned int>(i+4) || av[i].a_c != static_cast<unsigned int>(i+4))
        {
            printf("a_c data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%d] Actual av : [%d]\n", i+4, vec[i].a_c, av[i].a_c);
            return runall_fail;
        }

        if(vec[i].b_a != (i+5) || av[i].b_a != (i+5))
        {
            printf("b_a data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%d] Actual av : [%d]\n", i+5, vec[i].b_a, av[i].b_a);
            return runall_fail;
        }

        if(!AreAlmostEqual(vec[i].b_b, static_cast<float>(i+6)) || !AreAlmostEqual(av[i].b_b, static_cast<float>(i+6)))
        {
            printf("b_b data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%f] Actual av : [%f]\n", i+6, vec[i].b_b, av[i].b_b);
            return runall_fail;
        }

        if(vec[i].b_c != static_cast<unsigned int>(i+7) || av[i].b_c != static_cast<unsigned int>(i+7))
        {
            printf("b_c data copied to vector doesnt contained updated data for index : [%d]. FAIL!\n", i);
            printf("Expected: [%d] Actual vec: [%d] Actual av : [%d]\n", i+7, vec[i].b_c, av[i].b_c);
            return runall_fail;
        }

    }

    printf("PASS!\n");
    return runall_pass;
}

