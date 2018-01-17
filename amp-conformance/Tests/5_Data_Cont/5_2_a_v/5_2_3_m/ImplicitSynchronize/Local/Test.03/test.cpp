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
/// <summary>Create an AV, create a nested section and use that to write data</summary>

#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<int> v(100);
    Fill(v);
    array_view<int, 1> av(extent<1>(100), v);

    std::vector<int> random_data(v.size());
    Fill(v);

    // create a new read/write av and write to that
    array_view<int, 1> other = av.section(av.get_extent());
    for (unsigned int i = 0; i < static_cast<unsigned int>(v.size()); i++)
    {
        other[i] = random_data[i];
    }

    return VerifyDataOnCpu(av, random_data) ? runall_pass : runall_fail;
}
