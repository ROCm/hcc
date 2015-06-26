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
/// <summary>Use an index<N> to retrieve edge-values of an Array View<N></summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    ArrayViewTest<const int, 3> t(extent<3>(2, 3, 4));

    // set values through the underlying data pointer (since this is const)
    t.set_value(index<3>(0, 0, 0), 1);
    t.set_value(index<3>(0, 1, 3), 7);
    t.set_value(index<3>(1, 2, 3), 23);
    return
        t.view()[index<3>(0, 0, 0)] == 1 &&
        t.view()[index<3>(0, 1, 3)] == 7 &&
        t.view()[index<3>(1, 2, 3)] == 23
        ? t.pass() : t.fail();
}

