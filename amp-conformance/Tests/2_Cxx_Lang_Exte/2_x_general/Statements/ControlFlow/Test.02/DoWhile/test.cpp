// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Control Flow test: test nested code blocks with shared memory. Test with 2D forall</summary>

#include <iostream>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

const int XSize      = 8;
const int YSize      = 8;
const int Size       = XSize * YSize;

const int XGroupSize = 4;
const int YGroupSize = 4;

const int NumXGroups = XSize / XGroupSize;           // Make sure that Size is divisible by GroupSize
const int NumYGroups = YSize / YGroupSize;           // Make sure that Size is divisible by GroupSize
const int NumGroups  =  NumXGroups * NumYGroups;     // Make sure that Size is divisible by GroupSize

template<typename ElementType>
void CalculateGroupSum(ElementType* A, ElementType* B)
{
    int g = 0;
    for(int y = 0; y < YSize; y += YGroupSize)
    {
        for(int x = 0; x < XSize; x += XGroupSize)
        {
            // x,y is now the origin of the next group
            B[g] = (ElementType) 0;

            // calculate sum
            for(int gy = y; gy < (y + YGroupSize); gy++)
            {
                for(int gx = x; gx < (x + XGroupSize); gx++)
                {
                    int flatLocalIndex = gy * XSize + gx;
                    B[g] += A[flatLocalIndex];
                }
            }
            g++;
        }
    }
}

//Calculate sum of all elements in a group - GPU version
template<typename ElementType>
void CalculateGroupSum(tiled_index<YGroupSize, XGroupSize> idx, int flatLocalIndex, const array<ElementType, 2> & fA, array<ElementType, 2> & fB) __GPU_ONLY
{
    // use shared memory
    tile_static ElementType shared[XGroupSize * YGroupSize];
    shared[flatLocalIndex] = fA[idx.global];
    idx.barrier.wait();

    if(flatLocalIndex == 0)
    {
        ElementType sum = 0;
        for(int i = 0; i < XGroupSize * YGroupSize; i++)
        {
            sum += shared[i];
        }

        fB[idx.tile] = sum;
    }
}

template <typename ElementType>
void kernel(tiled_index<YGroupSize, XGroupSize> idx, const array<ElementType, 2> & fA, array<ElementType, 2> & fB, int x) __GPU_ONLY
{
    int flatLocalIndex = idx.local[0] * XGroupSize + idx.local[1];

    // Initialize to some fixed value; to check path when conditions are not true.
    // Only first thread initializes
    if(flatLocalIndex == 0) fB[idx.tile] = 100;

    do { if(x <= 1)  break; do { if(x <= 2)  break; do { if(x <= 3)  break; do { if(x <= 4)  break;
    do { if(x <= 5)  break; do { if(x <= 6)  break; do { if(x <= 7)  break; do { if(x <= 8)  break;
    do { if(x <= 9)  break; do { if(x <= 10) break; do { if(x <= 11) break; do { if(x <= 12) break;
    do { if(x <= 13) break; do { if(x <= 14) break; do { if(x <= 15) break; do { if(x <= 16) break;
    do { if(x <= 17) break; do { if(x <= 18) break; do { if(x <= 19) break; do { if(x <= 20) break;
    do {

        CalculateGroupSum<ElementType>(idx, flatLocalIndex, fA, fB);

        break;} while(true);
        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
        break;} while(true); break;} while(true); break;} while(true); break;} while(true);
}

template <typename ElementType>
runall_result test()
{
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

    Concurrency::extent<2> extentA(XSize, YSize), extentB(NumYGroups, NumXGroups);
    array<ElementType, 2> fA(extentA, rv), fB(extentB, rv);

    //forall where conditions are met
    copy(A, fA);
    int x = 30;
    parallel_for_each(fA.get_extent().template tile<YGroupSize, XGroupSize>(), [&, x] (tiled_index<YGroupSize, XGroupSize> idx) __GPU_ONLY {
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
    parallel_for_each(fA.get_extent().template tile<YGroupSize, XGroupSize>(), [&, x] (tiled_index<YGroupSize, XGroupSize> idx) __GPU_ONLY {
        kernel<ElementType>(idx, fA, fB, x);
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

