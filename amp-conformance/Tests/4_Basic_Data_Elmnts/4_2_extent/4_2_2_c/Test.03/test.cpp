// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create a new extent using an already initialized extent of the same rank and ensure that dimensions are copied correctly.</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    extent<1> e1(100);
    extent<1> e1n(e1);
    extent<2> e2(100, 200);
    extent<2> e2n(e2);
    extent<3> e3(100, 200, 300);
    extent<3> e3n(e3);

    if ((e1.rank != 1) || (e1[0] != e1n[0]))
    {
        return 11;
    }

    if ((e2.rank != 2) || (e2[0] != e2n[0]) || (e2[1] != e2n[1]))
    {
        return 12;
    }

    if ((e3.rank != 3) || (e3[0] != e3n[0]) || (e3[1] != e3n[1]) || (e3[2] != e3n[2]))
    {
        return 13;
    }

    const int cnt = 111;
    int data[cnt];

    for (int i = 0; i < cnt; i++)
    {
        data[i] = i;
    }

    extent<cnt> e(data);
    extent<cnt> en(e);

    if (en.rank != cnt)
    {
        return 14;
    }

    for (int i = 0; i < cnt; i++)
    {
        if (en[i] != data[i])
        {
            return 15;
        }
    }

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
    int result = test();

    printf("Test %s on host\n", ((result == 0) ? "passed" : "failed"));
    if(result != 0) return result;

    result = test_device();
    printf("Test %s on device\n", ((result == 0) ? "passed" : "failed"));
    return result;
}

