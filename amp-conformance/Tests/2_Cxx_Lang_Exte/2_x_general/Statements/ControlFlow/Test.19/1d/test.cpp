// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Call group_barrier with meaningless statement. For example, group_barrier is used without
/// shared memory and called directly. The program can be compiled and run. 1d</summary>

#include <iostream>
#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

const int GroupSize = 10;
const int NumGroups = 65;
const int Size      = GroupSize * NumGroups;

//Calculate sum of all elements in a group - GPU version
template<typename ElementType>
void CalculateGroupSum(tiled_index<GroupSize> idx, int flatLocalIndex, const array<ElementType, 1> & fA, array<ElementType, 1> & fB) __GPU_ONLY
{
    // meaningless
    idx.barrier.wait();

    if (idx.local[0] == 0)
        fB[idx.tile] = fA[idx];
}

//Kernel
template <typename ElementType>
void kernel(tiled_index<GroupSize> idx, const array<ElementType, 1> & fA, array<ElementType, 1> & fB, int x, int only1stgrp = 0) __GPU_ONLY
{
    if (only1stgrp == 0)
    {
        int flatLocalIndex = idx.local[0];

        // Initialize to some fixed value; to check path when conditions are not true.
        // Only first thread initializes
        if(flatLocalIndex == 0) fB[idx.tile] = 100;

        do { if(x <= 1)  break; do { if(x <= 2)  break; do { if(x <= 3)  break; do { if(x <= 4)  break; do { if(x <= 5)  break;
        for(;x > 6;)   { for(;x > 7;)  { for(;x > 8;)  { for(;x > 9;)  { for(;x > 10;) {
            if(x > 11) if(x > 12) if(x > 13) if(x > 14) if(x > 15)
            {
                switch(x > 16? 1:0) { case 0: break; case 1:
                    switch(x > 17? 1:0) { case 0: break; case 1: switch(x > 18? 1:0) { case 0: break; case 1:
                    switch(x > 19? 1:0) { case 0: break; case 1: switch(x > 20? 1:0) { case 0: break; case 1:
                {
                    while(x > 21) { while(x > 22) { while(x > 23) { while(x > 24) { while(x > 25){

                        CalculateGroupSum(idx, flatLocalIndex, fA, fB);

                        break;} break;} break;} break;} break;}

                }
                }}}}}
            }
            break;} break;} break;} break;} break;}
        break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);
    } else
    {
        if (idx.tile[0] == 0)
        {
            int flatLocalIndex = idx.local[0];

            // Initialize to some fixed value; to check path when conditions are not true.
            // Only first thread initializes
            if(flatLocalIndex == 0) fB[idx.tile] = 100;

            do { if(x <= 1)  break; do { if(x <= 2)  break; do { if(x <= 3)  break; do { if(x <= 4)  break; do { if(x <= 5)  break;
            for(;x > 6;)   { for(;x > 7;)  { for(;x > 8;)  { for(;x > 9;)  { for(;x > 10;) {
                if(x > 11) if(x > 12) if(x > 13) if(x > 14) if(x > 15)
                {
                    switch(x > 16? 1:0) { case 0: break; case 1:
                        switch(x > 17? 1:0) { case 0: break; case 1: switch(x > 18? 1:0) { case 0: break; case 1:
                        switch(x > 19? 1:0) { case 0: break; case 1: switch(x > 20? 1:0) { case 0: break; case 1:
                    {
                        while(x > 21) { while(x > 22) { while(x > 23) { while(x > 24) { while(x > 25){

                            CalculateGroupSum(idx, flatLocalIndex, fA, fB);

                            break;} break;} break;} break;} break;}

                    }
                    }}}}}
                }
                break;} break;} break;} break;} break;}
            break;} while(true); break;} while(true); break;} while(true); break;} while(true); break;} while(true);

        }
    }
}

template <typename ElementType>
runall_result test()
{
    srand(2012);
    bool passed = true;

    //ElementType A[Size]; // data
    ElementType *A = new ElementType[Size];
    ElementType *B = new ElementType[NumGroups];   // holds the grouped sum of data

    ElementType *refB2 = new ElementType[NumGroups]; // Expected value if the conditions are not satisfied. Some fixed values

    //Init A
    Fill<ElementType>(A, Size, 0, 100);

    int g = 0;
    for (int x = 0; x < Size; x += GroupSize)
    {
        refB2[g] = A[x];
        g++;
    }

    accelerator device = require_device_with_double(Device::ALL_DEVICES);

    accelerator_view rv = device.get_default_view();

    Concurrency::extent<1> extentA(Size), extentB(NumGroups);
    array<ElementType, 1> fA(extentA, rv), fB(extentB, rv);

    //forall where conditions are met
    copy(A, fA);

    int x = 26;
    parallel_for_each(extentA.tile<GroupSize>(), [&, x] (tiled_index<GroupSize> idx) __GPU_ONLY {
        kernel<ElementType>(idx, fA, fB, x);
    });

    copy(fB, B);

    if(!Verify<ElementType>(B, refB2, NumGroups))
    {
        passed = false;
        std::cout << "Test1: failed" << std::endl;
    }
    else
    {
        std::cout << "Test1: passed" << std::endl;
    }

    delete []A;
    delete []B;
    delete []refB2;

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

