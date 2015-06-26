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
/// <summary>Verify that .data is not allowed on rank > 1 array_view </summary>
//#Expects: Error: error C2338

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    const int m = 100, n = 80;
    const int size = m * n;

    vector<int> vec(size);

    array_view<int, 2> av(m, n, vec);

    // verify data
    equal(vec.begin(), vec.end(), av.data());

    return runall_fail;
}

