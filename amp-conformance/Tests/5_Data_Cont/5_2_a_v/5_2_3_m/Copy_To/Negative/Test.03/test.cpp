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
/// <summary>Copy an array_view to another array_view with incompatible element-type</summary>
//#Expects: Error: test.cpp\(44\) : error C2664

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int main()
{
    extent<1> src_ex(40);
    vector<long> src_vec(src_ex.size());
    Fill<long>(src_vec.data(), src_ex.size());
    array_view<long, 1> src(src_ex, src_vec);

    // same size, different element
    extent<1> dest_ex(src_ex.size());
    vector<int> dest_vec(dest_ex.size());
    array_view<int, 1> dest(dest_ex, dest_vec);

    // this should not compile
    src.copy_to(dest);
    return runall_fail;
}

