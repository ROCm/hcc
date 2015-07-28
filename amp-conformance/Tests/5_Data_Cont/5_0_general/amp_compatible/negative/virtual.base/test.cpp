// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create array with type that has virtual base.</summary>
//#Expects: Error: error C3581
//#Expects: Error: test.cpp\(38\)
//#Expects: Error: test.cpp\(41\)

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

struct A_base
{
    int m1;

public:
    int get()
    {
        return m1;
    }
};

struct A : virtual A_base
{

};

runall_result test_main()
{
    array<A, 1> arr(10);

    vector<A> vec(10);
    array_view<A, 1> arr_view(10, vec);

    return runall_fail;
}

