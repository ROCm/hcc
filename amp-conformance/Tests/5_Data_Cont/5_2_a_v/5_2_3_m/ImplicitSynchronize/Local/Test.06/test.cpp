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
/// <summary>Create an AV, create a section, reshape it and use that to write data</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;


int main()
{
    ArrayViewTest<long, 1> av(extent<1>(50));
    ArrayViewTest<long, 2, 1> reshaped = av.section(index<1>(10), extent<1>(30)).view_as(extent<2>(3, 10));

    reshaped.view()[index<2>(2, 2)] = 13;
    reshaped.set_known_value(index<2>(2, 2), 13);
    return av.view()[index<1>(32)] == 13 ? av.pass() : av.fail();
}