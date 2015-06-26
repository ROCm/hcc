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
/// <summary>View an AV as a shorter 1D AV</summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<int, 1> original(extent<1>(15));

    array_view<int, 1> shorter = original.view().view_as(extent<1>(10));

    // read and write between the original and higher rank view
    original.view()[index<1>(9)] = 17;
    original.set_known_value(index<1>(9), 17);

    shorter[index<1>(8)] = shorter[9];
    original.set_known_value(index<1>(8), 17);

    return
        original.view()[index<1>(8)] == 17 &&
        shorter[index<1>(8)] == 17
        ? original.pass() : original.fail();
}

