// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>create array with static member which is an amp container</summary>
//#Expects: Error: error C2338
//#Expects: Error: error C2338
//
//#Expects: Error: test.cpp\(31\)
//#Expects: Error: test.cpp\(32\)
//

#include <amp_graphics.h>
#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace concurrency;
using namespace concurrency::graphics;
using namespace Concurrency::Test;

struct A3
{
    static texture<float, 1> **arr;
};

runall_result test_main()
{
    array<A3, 1> arr(10);
    array_view<A3, 1> arr_view(arr);

    return runall_fail;
}

