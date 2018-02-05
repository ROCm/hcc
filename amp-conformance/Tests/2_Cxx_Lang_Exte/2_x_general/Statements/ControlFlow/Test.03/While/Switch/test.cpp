// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Control Flow test: while do and switch statement</summary>

#include <iostream>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

const int Size      = 32;
const int GroupSize = 8;

const int NumGroups = Size / GroupSize;     // Make sure that Size is divisible by GroupSize

//Calculate sum of all elements in a group - CPU version
template<typename ElementType>
void CalculateGroupSum(ElementType* A, ElementType* B)
{
    for(int g = 0; g < NumGroups; g++)
    {
        B[g] = (ElementType) 0;

        int groupOffset = g * GroupSize;
        for(int x = 0; x < GroupSize; x++)
        {
            int flatIndex = groupOffset + x;
            B[g] += A[flatIndex];
        }
    }
}

//Calculate sum of all elements in a group - GPU version
template<typename ElementType>
void CalculateGroupSum(tiled_index<GroupSize> idx, int flatLocalIndex, const array<ElementType, 1> & fA, array<ElementType, 1> & fB) __GPU_ONLY
{
    // use shared memory
    tile_static ElementType shared[GroupSize];
    shared[flatLocalIndex] = fA[idx.global];
    idx.barrier.wait();

    // first thread sums up the values of the group
    if(flatLocalIndex == 0)
    {
        ElementType sum = 0;
        for(int i = 0; i < GroupSize; i++)
        {
            sum += shared[flatLocalIndex + i];
        }

        fB[idx.tile] = sum;
    }
}

//Kernel
template <typename ElementType>
void kernel(tiled_index<GroupSize> idx, const array<ElementType, 1> & fA, array<ElementType, 1> & fB, int x) __GPU_ONLY
{
    int flatLocalIndex = idx.local[0];

    // Initialize to some fixed value; to check path when conditions are not true.
    // Only first thread initializes
    if(flatLocalIndex == 0) fB[idx.tile] = 100;

    while(x > 1)  { while(x > 2)  { while(x > 3)  { while(x > 4)  { while(x > 5) {
        while(x > 6)  { while(x > 7)  { while(x > 8)  { while(x > 9)  { while(x > 10){
            switch(x > 11? 1:0) { case 0: break; case 1: switch(x > 12? 1:0) { case 0: break; case 1:
                switch(x > 13? 1:0) { case 0: break; case 1: switch(x > 14? 1:0) { case 0: break; case 1:
                switch(x > 15? 1:0) { case 0: break; case 1: switch(x > 16? 1:0) { case 0: break; case 1:
                switch(x > 17? 1:0) { case 0: break; case 1: switch(x > 18? 1:0) { case 0: break; case 1:
                switch(x > 19? 1:0) { case 0: break; case 1: switch(x > 20? 1:0) { case 0: break; case 1:

                CalculateGroupSum<ElementType>(idx, flatLocalIndex, fA, fB);

            }}}}}}}}}}
            break;} break;} break;} break;} break;}
        break;} break;} break;} break;} break;}
}

template <typename ElementType>
runall_result test()
{
    srand(2012);
    bool passed = true;

    ElementType A[Size]; // data
    ElementType B[NumGroups];   // holds the grouped sum of data

    ElementType refB1[NumGroups]; // Expected value if conditions are satisfied; sum of elements in each group
    ElementType refB2[NumGroups]; // Expected value if the conditions are not satisfied. Some fixed values

    //Init A
    Fill<ElementType>(A, Size, 0, 100);

    //Init expected values
    CalculateGroupSum<ElementType>(A, refB1);

    for(int g = 0; g < NumGroups; g++)
    {
        refB2[g] = 100; // Init to fixed value
    }

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    Concurrency::extent<1> extentA(Size), extentB(NumGroups);
    array<ElementType, 1> fA(extentA, rv), fB(extentB, rv);

    //forall where conditions are met
    copy(A, fA);
    int x = 100;
    parallel_for_each(extentA.tile<GroupSize>(), [&, x] (tiled_index<GroupSize> idx) __GPU_ONLY {
        kernel<ElementType>(idx, fA, fB, x);
    });

    copy(fB, B);

    if(!Verify<ElementType>(B, refB1, NumGroups))
    {
        passed = false;
        std::cout << "Test1: failed" << std::endl;
    }
    else
    {
        std::cout << "Test1: passed" << std::endl;
    }

    //forall where conditions are not met
    copy(A, fA);
    x = 5;
    parallel_for_each(extentA.tile<GroupSize>(), [&,x] (tiled_index<GroupSize> idx) __GPU_ONLY {
        kernel<ElementType>(idx, fA, fB, 10);
    });

    copy(fB, B);

    if(!Verify<ElementType>(B, refB2, NumGroups))
    {
        passed = false;
        std::cout << "Test2: " << "Failed!" << std::endl;
    }
    else
    {
        std::cout << "Test2: passed" << std::endl;
    }

    return passed;
}


runall_result test_main()
{
    runall_result result;

    std::cout << "Test shared memory with \'int\'" << std::endl;
    result = test<int>();
    if(result != runall_pass) return result;

    std::cout << "Test shared memory with \'unsigned int\'" << std::endl;
    result = test<unsigned int>();
    if(result != runall_pass) return result;

    std::cout << "Test shared memory with \'long\'" << std::endl;
    result = test<long>();
    if(result != runall_pass) return result;

    std::cout << "Test shared memory with \'unsigned long\'" << std::endl;
    result = test<unsigned long>();
    if(result != runall_pass) return result;

    std::cout << "Test shared memory with \'float\'" << std::endl;
    result = test<float>();
    if(result != runall_pass) return result;

    std::cout << "Test shared memory with \'double\'" << std::endl;
    result = test<double>();
    if(result != runall_pass) return result;


    return result;
}

