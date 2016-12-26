// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Reinterpret an Array of 3 floats as float (CPU)</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;

class Foo
{
public:
    float r;
    float b;
    float g;
};

runall_result test_main()
{
    std::vector<float> v(10 * 3);
    Fill(v);

    array<Foo, 1> arr_rbg(10, reinterpret_cast<Foo *>(v.data()));
	array_view<Foo, 1> av_rbg(arr_rbg);

    array_view<float, 1> av_float = arr_rbg.reinterpret_as<float>();

    int expected_size = arr_rbg.get_extent().size() * sizeof(Foo) / sizeof(float);
    Log(LogType::Info, true) << "Expected size: " << expected_size << " actual: " << av_float.get_extent()[0] << std::endl;
    if (av_float.get_extent()[0] != expected_size)
    {
        return runall_fail;
    }

    return Verify<float>(reinterpret_cast<float *>(av_rbg.data()), av_float.data(), expected_size) ? runall_pass : runall_fail;
}


