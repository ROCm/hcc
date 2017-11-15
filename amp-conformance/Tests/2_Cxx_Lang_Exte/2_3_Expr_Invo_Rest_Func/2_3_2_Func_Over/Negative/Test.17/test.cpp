// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P2</tags>
/// <summary>(Negative) If calling less restricted function, all candidate
///  functions must be provided.</summary>
//#Expects: Error: error C3930

#include <amptest.h>

using std::vector;
using namespace Concurrency;
using namespace Concurrency::Test;

static
inline
int f() restrict(amp)
{
    return 0;
}

static
inline
int f(float) restrict(amp)
{
    return 1;
}

int f() restrict(cpu)
{
    return 0;
}

int b1(float x) restrict(amp, cpu)
{
    return f(x);
}

int b2(float f) restrict(amp, cpu)
{
    return b1(f);
}

int b3(float f) restrict(amp, cpu)
{
    return b2(f);
}

int p(float f) restrict(amp)
{
    return b3(f);
}

int test()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    vector<int> vA(1, 0);
    array<int, 1> aA(extent<1>(1), vA.begin(), rv);

    parallel_for_each(aA.get_extent(), [&](index<1> idx) restrict(amp){
        float f = 0.0;

        aA[idx] = p(f);
    });

    vA = aA;

    return ((vA[0] == 1) ? 0 : 1);
}

int main(int argc, char **argv)
{
    int ret = test();

    printf("failed\n");

    return 1;
}

