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
/// <summary>Create and destroy an array_view in a loop. Ensure that resources are not leaked
/// Declare the underlying array outside the loop, declare the array_view inside the loop</summary>


#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
   const int size = 10000;
   const int iterations = 1000;

   vector<int> vec(size);
   for(int i = 0; i < size;i++)
   {
        vec[i] = i;
   }

   extent<1> ex(size);
   array<int, 1> arr(ex, vec.begin(), accelerator(accelerator::cpu_accelerator).get_default_view());


   for(int i = 0; i < iterations; i++)
   {
        //create array_view inside the loop
        array_view<int, 1> av(arr);

        parallel_for_each(av.get_extent(), [=](index<1> idx) __GPU
        {
            av[idx]++;
        });

        av.synchronize();
   }

   //verify
   for(int i = 0; i < size;i++)
   {
        if(arr[i] != (i + iterations))
        {
             printf("Fail: array element doesnt match expected value. Expected: [%d] Actual: [%d]\n", i + iterations, arr[i]);
             return runall_fail;
        }
   }

   printf("Pass!");
   return runall_pass;
}
