// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Test user defined converison functions with __GPU context.</summary>

#include "amptest.h"

using namespace concurrency;
using namespace concurrency::Test;

class c2 {};

class c1
{
public:
    c1() : m(0) {}
    c1() __GPU_ONLY : m(0) {}

    operator c2() __GPU_ONLY
    {
        c2 o;

        m = 1;
        return o;
    }

    operator c2()
    {
        c2 o;

        m = 2;
        return o;
    }

    int m;
};

runall_result test() __GPU
{
    c1 obj;
    c2 obj2 = obj;

    // this value would be 2 if the CPU version were called
    return obj.m == 1;
}

int main()
{
    return GPU_INVOKE(require_device(device_flags::NOT_SPECIFIED).get_default_view(), runall_result, test).get_exit_code();
}

