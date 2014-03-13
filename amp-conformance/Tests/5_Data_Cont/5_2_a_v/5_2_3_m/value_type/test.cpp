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
/// <summary>Verify array_view::value_type typedef</summary>
#include <amptest.h>
#include <amptest_main.h>
#include <type_traits>
using namespace concurrency;

class UDT { int i; float f; };

#define VERIFY(T, N) static_assert(std::is_same<array_view<T, N>::value_type, T>::value, "static_assert failed for " #T);

VERIFY(int, 1);
VERIFY(const int, 2);
VERIFY(float, 3);
VERIFY(const float, 4);
VERIFY(UDT, 5);
VERIFY(const UDT, 6);

