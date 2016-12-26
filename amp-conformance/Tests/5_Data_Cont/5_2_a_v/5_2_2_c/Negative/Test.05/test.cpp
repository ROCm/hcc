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
/// <summary>Verify that we cannot create an array_view with amp incompatible type and pass it to a parallel_for_each</summary>
//#Expects: Error: error C2338
//#Expects: Error: error C3581
//#Expects: Error: error C3581
//#Expects: Error: error C3581

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

struct s_a
{
    int a;
    long b;
};

template<typename T>
int test()
{
    const int size = 100;
    vector<T> v(size);
    array_view<T, 1> av(size, v);
}

int main()
{
   test<bool>();
   test<int*>();
   test<s_a*>();
   test<s_a>();
}

