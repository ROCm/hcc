// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Make sure we can create a tile_static variable of type empty class inside an amp function</summary>

#include <amptest.h>
#include <amptest_main.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct A
{
    int get() restrict(cpu, amp)
    {
        return 1;
    }
};

int test() restrict(amp)
{
    tile_static A a;

    return a.get();
}

runall_result test_main()
{
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    int out_ = 0;
    array_view<int> out(1, &out_);
    parallel_for_each(av, out.get_extent().tile<1>(), [=](tiled_index<1> tidx) restrict(amp)
    {
        out[tidx.global] = test();
    });

    return out[0] == 1;
}
