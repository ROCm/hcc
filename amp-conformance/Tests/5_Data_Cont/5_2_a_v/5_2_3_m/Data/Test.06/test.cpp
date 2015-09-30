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
/// <summary>Create and access the data pointer for a view with a struct</summary>

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

struct Foo
{
    int i;
    float f;
};

int main()
{
    const int size = 2;
    vector<Foo> vec(size);
    vec[0].i = 13;
    vec[0].f = 14.0;
    vec[1].i = 17;
    vec[1].f = 18.0;


    extent<1> ex(size);
    array_view<Foo> original(ex, vec);

    Foo *data = original.data();
    if (data[0].i == 13 && data[0].f == 14.0 && data[1].i == 17 && data[1].f == 18.0)
    {
        return runall_pass;
    }
    else
    {
        return runall_fail;
    }
}

