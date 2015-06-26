// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>(Negative) Assign an initialized extent with a different rank to this extent and ensure that compilation fails.</summary>
//#Expects: Error: test.cpp\(28\) : error C2679
//#Expects: Error: test.cpp\(33\) : error C2679
//#Expects: Error: test.cpp\(38\) : error C2679
//#Expects: Error: test.cpp\(43\) : error C2679
//#Expects: Error: test.cpp\(48\) : error C2679
//#Expects: Error: test.cpp\(53\) : error C2679

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    extent<1> e1a;
    extent<2> e2a;

    e2a = e1a;

    extent<1> e1b(100);
    extent<2> e2b(100, 200);

    e2b = e1b;

    extent<3> e3a;
    extent<4> e4a;

    e4a = e3a;

    extent<3> e3b(100, 200, 300);
    extent<4> e4b;

    e4b = e3b;

    extent<4> e4c;
    extent<5> e5c;

    e5c = e4c;

    extent<9> e9a;
    extent<99> e99a;

    e99a = e9a;

    return 1;
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

    parallel_for_each(e, [&](index<1> idx) __GPU {
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

    return true;
}

int main()
{
    int result = test();

    printf("Test %s on host\n", ((result == 0) ? "passed" : "failed"));
    if(result != 0) return result;

    result = test_device();
    printf("Test %s on device\n", ((result == 0) ? "passed" : "failed"));
    return result;
}

