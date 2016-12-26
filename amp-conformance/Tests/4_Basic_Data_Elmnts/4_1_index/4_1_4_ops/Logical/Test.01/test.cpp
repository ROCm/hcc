// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Compare two index objects using the logic operator</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test_equal() __GPU
{
    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-10, -1, 0, 1, 10};

    index<5> i1(data1);
    index<5> i2(data2);

    if (!(i1 == i2))
    {
        return 11;  // test_equal scenario1 failed
    }

    int data3[] = {-10, -1, 0, 1, 10};
    int data4[] = {-101, -11, 1, 11, 101};

    index<5> i3(data3);
    index<5> i4(data4);

    if (i3 == i4)
    {
        return 12;  // test_equal scenario2 failed
    }

    int data5[] = {1, 2, 3, 4, 5};
    int data6[] = {11, 2, 3, 4, 5};

    index<5> i5(data5);
    index<5> i6(data6);

    if (i5 == i6)
    {
        return 13;  // test_equal scenario3 failed
    }

    int data7[] = {1, 2, 3, 4, 5};
    int data8[] = {1, 2, 3, 4, 55};

    index<5> i7(data7);
    index<5> i8(data8);

    if (i7 == i8)
    {
        return 14;  // test_equal scenario4 failed
    }

    int data9[] = {1, 2, 3, 4, 5};
    int data10[] = {1, 22, 33, 44, 55};

    index<5> i9(data9);
    index<5> i10(data10);

    if (i9 == i10)
    {
        return 15;  // test_equal scenario5 failed
    }

    int data11[] = {1, 2, 3, 4, 5};
    int data12[] = {11, 22, 33, 44, 5};

    index<5> i11(data11);
    index<5> i12(data12);

    if (i11 == i12)
    {
        return 16;  // test_equal scenario6 failed
    }

    return 0;
}

int test_not_equal() __GPU
{
    int data1[] = {-10, -1, 0, 1, 10};
    int data2[] = {-100, -11, 1, 11, 100};

    index<5> i1(data1);
    index<5> i2(data2);

    if (!(i1 != i2))
    {
        return 21;  // test_not_equal scenario1 failed
    }

    int data3[] = {-10, -1, 0, 1, 10};
    int data4[] = {-10, -1, 0, 1, 10};

    index<5> i3(data3);
    index<5> i4(data4);

    if (i3 != i4)
    {
        return 22;  // test_not_equal scenario2 failed
    }

    int data5[] = {-10, -1, 0, 1, 10};
    int data6[] = {-10, -11, 2, 11, 101};

    index<5> i5(data5);
    index<5> i6(data6);

    if (!(i5 != i6))
    {
        return 23;  // test_not_equal scenario3 failed
    }

    int data7[] = {-10, -1, 0, 1, 10};
    int data8[] = {-101, -11, 1, 11, 10};

    index<5> i7(data7);
    index<5> i8(data8);

    if (!(i7 != i8))
    {
        return 24;  // test_not_equal scenario4 failed
    }

    int data9[] = {-10, -1, 0, 1, 10};
    int data10[] = {-100, -1, 0, 1, 10};

    index<5> i9(data9);
    index<5> i10(data10);

    if (!(i9 != i10))
    {
        return 25;  // test_not_equal scenario5 failed
    }

    int data11[] = {-10, -1, 0, 1, 10};
    int data12[] = {-10, -1, 0, 1, 100};

    index<5> i11(data11);
    index<5> i12(data12);

    if (!(i11 != i12))
    {
        return 26;  // test_not_equal scenario6 failed
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
    accelerator_view av = require_device(device_flags::NOT_SPECIFIED).get_default_view();

    Concurrency::extent<1> e(size);
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
            int ret = presult[i];
            return ret;
        }
    }

    return 0;
}

runall_result test_main()
{
	runall_result result;
    // Test on host
    result &= REPORT_RESULT(test() == 0);

    // Test on device
	result &= REPORT_RESULT(test_device() == 0);
    return result;
}


