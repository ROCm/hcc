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
/// <summary>Reinterpret an AV of float as double (CPU)</summary>

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    std::vector<float> v(10);
    Fill(v);

    array_view<float, 1> av_float(static_cast<int>(v.size()), v);
    array_view<double, 1> av_double = av_float.reinterpret_as<double>();

    int expected_size = av_float.get_extent().size() * sizeof(float) / sizeof(double);
    Log(LogType::Info, true) << "Expected size: " << expected_size << " actual: " << av_double.get_extent()[0] << std::endl;
    if (av_double.get_extent()[0] != expected_size)
    {
        return runall_fail;
    }

    return Verify<double>(reinterpret_cast<double *>(av_float.data()), av_double.data(), expected_size) ? runall_pass : runall_fail;
}

