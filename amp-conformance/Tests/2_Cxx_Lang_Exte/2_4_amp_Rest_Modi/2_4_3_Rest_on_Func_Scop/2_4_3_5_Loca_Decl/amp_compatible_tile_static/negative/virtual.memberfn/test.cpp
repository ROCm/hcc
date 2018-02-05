// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create local variable with type that has virtual function.</summary>
//#Expects: Error: test.cpp\(40\) : error C3581
//#Expects: Error: test.cpp\(23\)
//#Expects: Error: test.cpp\(28\)

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

class A_base
{
    int m1;

    virtual int get() = 0;
};

struct A: A_base
{
    int get()
    {
        return 1;
    }
};

runall_result test_main()
{
    array<int, 1> arr(10);

    parallel_for_each(arr.get_extent(), [&](index<1> idx) restrict(amp)
    {
        tile_static A a[10];

        arr[idx] = a.get();
    });

    return runall_fail;
}

