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
/// <summary>Use an index<N> to get/set edge-values of an Array View<N> on the GPU</summary>

#include <amptest/array_view_test.h>
#include <amptest/coordinates.h>
#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

int main()
{
    accelerator accel = require_device(device_flags::NOT_SPECIFIED);

    ArrayViewTest<int, 2> t(extent<2>(3, 3));
    array_view<int, 2> av = t.view();

    Log(LogType::Info, true) << "Setting known values on the GPU" << std::endl;
    // now set some values on the GPU
    parallel_for_each(accel.get_default_view(), extent<1>(1), [av](index<1>) __GPU {
        av[index<2>(0, 0)] = 1;
        av[index<2>(2, 2)] = 19;
    });

    // use this to get the results
    std::vector<int> v_results(av.get_extent().size());
    array_view<int, 2> results(av.get_extent(), v_results);
    parallel_for_each(accel.get_default_view(), av.get_extent(), [av, results](index<2> i) __GPU {
        results[i] = av[i];
    });

    // update the tracking structure with the known-values
    t.set_known_value(index<2>(0, 0), 1);
    t.set_known_value(index<2>(2, 2), 19);

    return
        // verify the results av, calling pass() will verify the original view
        results[index<2>(0, 0)] == 1 &&
        results[index<2>(2, 2)] == 19 &&
        av[index<2>(0, 0)] == 1 &&
        av[index<2>(2, 2)] == 19
        ? t.pass() : t.fail();
}

