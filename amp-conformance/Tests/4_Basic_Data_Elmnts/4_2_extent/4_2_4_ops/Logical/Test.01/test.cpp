// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compare two extents using the logic operator (when compiled with /Od, very slow compilation)</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test_equal() __GPU
{
    int data1[] = {1, 2, 9999, 3, 2147483647};
    int data2[] = {1, 2, 9999, 3, 2147483647};

    extent<5> e1(data1);
    extent<5> e2(data2);

    if (!(e1 == e2))
    {
        return 11;
    }

    int data3[] = {10, 1, 4, 1, 10};
    int data4[] = {101, 11, 1, 11, 101};

    extent<5> e3(data3);
    extent<5> e4(data4);

    if (e3 == e4)
    {
        return 12;
    }

    int data5[] = {1, 2, 3, 4, 5};
    int data6[] = {11, 2, 3, 4, 5};

    extent<5> e5(data5);
    extent<5> e6(data6);

    if (e5 == e6)
    {
        return 13;
    }

    int data7[] = {1, 2, 3, 4, 5};
    int data8[] = {1, 2, 3, 4, 55};

    extent<5> e7(data7);
    extent<5> e8(data8);

    if (e7 == e8)
    {
        return 14;
    }

    int data9[] = {1, 2, 3, 4, 5};
    int data10[] = {1, 22, 33, 44, 55};

    extent<5> e9(data9);
    extent<5> e10(data10);

    if (e9 == e10)
    {
        return 15;
    }

    int data11[] = {1, 2, 3, 4, 5};
    int data12[] = {11, 22, 33, 44, 55};

    extent<5> e11(data11);
    extent<5> e12(data12);

    if (e11 == e12)
    {
        return 16;
    }

    return 0;
}

int test_not_equal() __GPU
{
    int data1[] = {10, 1, 5, 1, 10};
    int data2[] = {100, 11, 1, 11, 100};

    extent<5> e1(data1);
    extent<5> e2(data2);

    if (!(e1 != e2))
    {
        return 21;
    }

    int data3[] = {10, 1, 2, 1, 10};
    int data4[] = {10, 1, 2, 1, 10};

    extent<5> e3(data3);
    extent<5> e4(data4);

    if (e3 != e4)
    {
        return 22;
    }

    int data5[] = {1, 1, 1, 1, 1};
    int data6[] = {2, 1, 1, 1, 1};

    extent<5> e5(data5);
    extent<5> e6(data6);

    if (!(e5 != e6))
    {
        return 23;
    }

    int data7[] = {1, 1, 1, 1, 1};
    int data8[] = {1, 1, 1, 1, 2};

    extent<5> e7(data7);
    extent<5> e8(data8);

    if (!(e7 != e8))
    {
        return 24;
    }

    int data9[] = {1, 1, 1, 1, 1};
    int data10[] = {1, 1, 2, 1, 1};

    extent<5> e9(data9);
    extent<5> e10(data10);

    if (!(e9 != e10))
    {
        return 25;
    }

    int data11[] = {1, 1, 1, 1, 1};
    int data12[] = {2, 2, 2, 2, 2};

    extent<5> e11(data11);
    extent<5> e12(data12);

    if (!(e11 != e12))
    {
        return 26;
    }

    return 0;
}

int test() __GPU
{
    int result = test_equal();
    if(result != 0)
    {
        return result;
    }

    return test_not_equal();
}

void kernel(index<1>& idx, array<int, 1>& result) __GPU
{
    result[idx] = test();
}

const int size = 10;

int test_device()
{
    accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        return 2;
    }
    accelerator_view av = device.get_default_view();

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

    return 0;
}

int main(int argc, char **argv)
{
   int result = test();

    printf("Test %s on host\n", ((result == 0) ? "passed" : "failed"));
    if(result != 0) return result;

    result = test_device();
    printf("Test %s on device\n", ((result == 0) ? "passed" : "failed"));
    return result;

}

