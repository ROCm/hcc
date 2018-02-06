// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Create an extent using the default constructor. Ensure that indices for all dimensions are initialized to 0</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    extent<1> e1;

    if (e1[0] != 0)
    {
        return 11;
    }

    extent<2> e2;

    if ((e2[0] != 0) || (e2[1] != 0))
    {
        return 12;
    }

    extent<3> e3;

    if ((e3[0] != 0) || (e3[1] != 0) || (e3[2] != 0))
    {
        return 13;
    }

    extent<4> e4;

    if ((e4[0] != 0) || (e4[1] != 0) || (e4[2] != 0) || (e4[3] != 0))
    {
        return 14;
    }

    extent<10> e10;

    for(int i = 0; i < 10; i ++)
    {
        if (e10[i] != 0)
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

    parallel_for_each(e, [&](index<1> idx) __GPU {
        kernel(idx, result);
    });
    presult = result;;

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

