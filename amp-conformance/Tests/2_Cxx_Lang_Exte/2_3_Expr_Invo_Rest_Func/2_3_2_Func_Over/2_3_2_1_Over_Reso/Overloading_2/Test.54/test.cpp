// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>Can call more restricted function</summary>

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

class c2
{
public:
    int f(int) restrict(amp, cpu)
    {
        return 0;
    }

    int f(float) restrict(amp, cpu)
    {
        return 1;
    }
};

class c1
{
public:
    int b(int) restrict(amp)
    {
        c2 o;

        float i = 1.0f;

        return o.f(i);
    }
};

int test(accelerator_view &rv)
{
    extent<1> e(1);
    vector<int> vA(1, 0);
    array<int, 1> aA(e, vA.begin(), vA.end(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) restrict(amp) {
        c1 o;

        aA[idx] = o.b(1);
    });

    vA = aA;

    return ((vA[0] == 1) ? 0 : 1);
}

int main(int argc, char **argv)
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();
    int ret = test(rv);

    printf("%s\n", (ret == 0)? "passed\n" : "failed");

    return (ret == 0) ? 0 : 1;
}

