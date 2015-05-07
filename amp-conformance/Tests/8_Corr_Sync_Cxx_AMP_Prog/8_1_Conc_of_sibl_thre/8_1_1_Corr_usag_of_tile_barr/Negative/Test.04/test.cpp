// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Call barrier with non group forall call. Compile time error is prompted</summary>

#include <iostream>
#include <amptest.h>
#include <amptest_main.h>
using namespace std;
using namespace Concurrency;
using namespace Concurrency::Test;

const int XGroupSize = 8;
const int YGroupSize = 8;
const int ZGroupSize = 16;

const int NumXGroups = 32;
const int NumYGroups = 32;
const int NumZGroups = 63;
const int NumGroups  =  NumXGroups * NumYGroups * NumZGroups;

const int XSize      = XGroupSize * NumXGroups;
const int YSize      = YGroupSize * NumYGroups;
const int ZSize     = ZGroupSize * NumZGroups;
const int Size = XSize * YSize * ZSize;

template <typename ElementType>
runall_result test()
{
    srand(2012);
    bool passed = true;

    //ElementType A[Size]; // data
    ElementType *A = new ElementType[Size];
    ElementType *B = new ElementType[NumGroups];   // holds the grouped sum of data

    ElementType *refB1 = new ElementType[NumGroups]; // Expected value if conditions are satisfied; sum of elements in each group
    ElementType *refB2 = new ElementType[NumGroups]; // Expected value if the conditions are not satisfied. Some fixed values

    //Init A
    Fill<ElementType>(A, Size, 0, 100);

    for(int g = 0; g < NumGroups; g++)
    {
        refB2[g] = 100; // Init to fixed value
    }

    accelerator_view rv =  require_device(Device::ALL_DEVICES).get_default_view();

    Concurrency::extent<3> extentA(ZSize, YSize, XSize), extentB(NumZGroups, NumYGroups, NumXGroups);
    Concurrency::array<ElementType, 3> fA(extentA, rv), fB(extentB, rv);

    //forall where conditions are met
    copy(A, fA);

    int x = 26;
    parallel_for_each(fA.get_extent(), [&, x] (index<3> idx) __GPU_ONLY {
        idx.barrier.wait();
    });

    copy(fB, B);

    if(!Verify<ElementType>(B, refB1, NumGroups))
    {
        passed = false;
        cout << "Test1: failed" << endl;
    }
    else
    {
        cout << "Test1: passed" << endl;
    }

    delete []A;
    delete []B;
    delete []refB1;
    delete []refB2;
    return passed;
}

runall_result test_main()
{
    runall_result result;

    cout << "Test shared memory with \'int\'" << endl;
    result = test<int>();


    return runall_fail;
}

//#Expects: Error: \(61\) : .+ C2039
//#Expects: Error: \(61\) : .+ C2039

