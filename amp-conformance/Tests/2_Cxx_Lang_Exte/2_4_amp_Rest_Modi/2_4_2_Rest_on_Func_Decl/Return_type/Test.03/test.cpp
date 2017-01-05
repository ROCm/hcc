// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Global function which returns a reference</summary>

#include "amptest.h"
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

int & test(int &in) __GPU
{
    return in;
}

int main()
{
    std::vector<int> v(1);
    array_view<int, 1> av(1, v);

    auto lam = [=](index<1> i) __GPU {
        int a = 15;
        av[0] = test(a);
    };
    parallel_for_each(extent<1>(1), lam);

    if (av[0] != 15)
    {
        Log(LogType::Info, true) << "Result was: " << av[0] << " Expected: 15" << std::endl;
        return runall_fail;
    }

    return runall_pass;
}
