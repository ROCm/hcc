// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once

#include <amptest.h>
#include <amptest_main.h>
#include <cstdint>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

#define INIT_VALUE ((int)0xABCDEF98)

// TOOD: Instead of these functions, use a type_comparer<T>.are_equal. It handles the 'almost equal' semantics for you.
template <typename T>
bool Equal(T in1, T in2) __GPU
{
    return Concurrency::Test::details::AreEqual(in1, in2);
}

template <>
bool Equal(float in1, float in2) __GPU
{
    return amptest_math::are_almost_equal(in1, in2);
}

template <>
bool Equal(double in1, double in2) __GPU
{
    return amptest_math::are_almost_equal(in1, in2);
}


