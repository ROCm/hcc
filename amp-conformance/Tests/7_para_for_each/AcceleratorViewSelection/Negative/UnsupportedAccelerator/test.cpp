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
/// <summary>parallel_for_each on an unsupported accelerator</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace concurrency;
using namespace Concurrency::Test;

static const char *errorMsg_UnsupportedAccelerator = "concurrency::parallel_for_each is not supported on the selected accelerator \"CPU accelerator\".";

bool Test1()
{
    srand(2010);
    const size_t size = 1024;
    int *results = new int[size];

    array<int, 1> arr(size, results, accelerator(accelerator::cpu_accelerator).get_default_view());

    try {
        parallel_for_each(extent<1>(size), [&](index<1> idx) __GPU {
            arr[idx] += 1;
        });

        return false;
    }
    catch(runtime_exception &e) {
        if (strstr(e.what(), errorMsg_UnsupportedAccelerator) != NULL) {
            return true;
        }
        else {
            return false;
        }
    }
}

bool Test2()
{
    srand(2010);
    const size_t size = 1024;
    int *results = new int[size];

    array<int, 1> arr(size, results, accelerator(accelerator::cpu_accelerator).get_default_view());

    try {
        parallel_for_each(accelerator(accelerator::cpu_accelerator).get_default_view(), extent<1>(size), [&](index<1> idx) __GPU {
            arr[idx] += 1;
        });

        return false;
    }
    catch(runtime_exception &e) {
        if (strstr(e.what(), errorMsg_UnsupportedAccelerator) != NULL) {
            return true;
        }
        else {
            return false;
        }
    }
}

bool Test3()
{
    srand(2010);
    const size_t size = 1024;
    int *results = new int[size];

    array<int, 1> arr(size, results, accelerator(accelerator::cpu_accelerator).get_default_view());

    try {
        parallel_for_each(accelerator(accelerator::cpu_accelerator).get_default_view(), extent<1>(size), [&](index<1> idx) __GPU {
            arr[idx] += 1;
        });

        return false;
    }
    catch(runtime_exception &e) {
        if (strstr(e.what(), errorMsg_UnsupportedAccelerator) != NULL) {
            return true;
        }
        else {
            return false;
        }
    }
}

runall_result test_main()
{
    runall_result result;

    result = REPORT_RESULT((Test1()));
    result = REPORT_RESULT((Test2()));
    result = REPORT_RESULT((Test3()));

    return result;
}

