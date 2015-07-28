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
/// <summary>Create an AV, create overlapping views and use both to write data</summary>

#include <amptest/array_view_test.h>
#include <amptest.h>
#include <vector>
#include <algorithm>

using namespace Concurrency;
using namespace Concurrency::Test;


int main()
{
    ArrayViewTest<long, 1> av(extent<1>(50));
    ArrayViewTest<long, 1> section1 = av.section(index<1>(10), extent<1>(30));
    ArrayViewTest<long, 1> section2 = av.section(index<1>(0), extent<1>(25));

    section1.view()[3] = 17;
    section1.set_known_value(index<1>(3), 17);
    section2.view()[13] = 19;
    section2.set_known_value(index<1>(13), 19);
    section2.view()[3] = 15;
    section2.set_known_value(index<1>(3), 15);

    return av.view()[13] == 19 && av.view()[3] == 15 ? av.pass() : av.fail();
}