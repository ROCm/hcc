// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Create a new extent using an already initialized extent of the different rank and ensure that compilation fails.</summary>
//#Expects: Error: error C2664

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    extent<1> e1(100);
    extent<2> e1n(e1);
    extent<2> e2(100, 200);
    extent<3> e2n(e2);
    extent<3> e3(100, 200, 300);
    extent<4> e3n(e3);
    extent<5> e4n(e3);

    return 0;
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator acc;
    if (!get_device(Device::ALL_DEVICES, acc))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view av = acc.get_default_view();

    extent<1> e(size);
    array<int, 1> result(e, av);
    vector<int> presult(size, 0);

    parallel_for_each(e, [&](index<1> idx) __GPU{
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < 10; i++)
    {
        if (presult[i] != 0)
        {
            printf("Test failed. Return code: %d\n", presult[i]);
            return 1;
        }
    }

    return 0;
}

int main()
{
    test();
    test_device();

    //Always fail if this succeeds to compile
    printf("Failed!\n");
    return 1;
}

