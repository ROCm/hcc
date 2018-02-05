// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Const and non-const restrict(amp) member function</summary>

#include "amptest.h"
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct S
{
    int test() const restrict(amp, cpu)
    {
        return 2;
    }

    int test() restrict(amp, cpu)
    {
        return 1;
    }
};

int main()
{
    std::vector<int> v(1);
    array_view<int, 1> av(1, v);

    Log(LogType::Info, true) << "Calling const function" << std::endl;
    parallel_for_each(extent<1>(1), [=](index<1> i) __GPU {
        const S s;
        av[0] = s.test();
    });

    if (av[0] != 2)
    {
        Log(LogType::Info, true) << "Result was: " << av[0] << " Expected: 2" << std::endl;
        return runall_fail;
    }

    Log(LogType::Info, true) << "Calling non-const function" << std::endl;
    parallel_for_each(extent<1>(1), [=](index<1> i) __GPU {
        S s;
        av[0] = s.test();
    });

    if (av[0] != 1)
    {
        Log(LogType::Info, true) << "Result was: " << av[0] << " Expected: 1" << std::endl;
        return runall_fail;
    }

    return runall_pass;
}
