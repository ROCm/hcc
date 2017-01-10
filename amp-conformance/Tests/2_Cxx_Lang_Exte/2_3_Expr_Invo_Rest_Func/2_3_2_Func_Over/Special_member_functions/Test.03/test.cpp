// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Restrict(amp, cpu) constructor, overloaded assignment and copy operators, generated destructor</summary>

#include "amptest.h"
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;

struct S
{
    S() : x(0), y(0) {}

    S(int x) __GPU
    : x(x)
    {
        y = 2 * x;
    }

    S(const S& other) __GPU_ONLY
    {
        this->x = other.x;
        this->y = other.y;
    }

    S(const S& other) __CPU_ONLY
    {
        this->x = 0;
        this->y = 0;
    }

    S& operator=(const S& other) __GPU_ONLY
    {
        this->x = other.x;
        this->y = other.y;
        return *this;
    }

    S& operator=(const S& other) __CPU_ONLY
    {
        this->x = 0;
        this->y = 0;
        return *this;
    }

    int x;
    int y;
};

int main()
{
    require_device(device_flags::NOT_SPECIFIED);

    std::vector<S> v(1);
    array_view<S, 1> av(1, v);

    Log(LogType::Info, true) << "Constructing S(12) on the GPU" << std::endl;
    parallel_for_each(extent<1>(1), [=](index<1> i) __GPU {
        S s1(12);
        S s2 = s1;
        S s3(s2);
        av[0] = s3;
    });

    if (av[0].x != 12 && av[0].y != 24)
    {
        Log(LogType::Info, true) << "Expected: { 12, 24 } Was: {" << av[0].x << ", " << av[0].y << "}" << std::endl;
        return runall_fail;
    }

    return runall_pass;
}
