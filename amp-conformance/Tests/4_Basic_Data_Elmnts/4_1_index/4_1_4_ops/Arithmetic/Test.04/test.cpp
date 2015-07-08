// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P1</tags>
/// <summary>Check that operator precedence. Test nesting and chaining of operators.</summary>

#include <amptest.h>
#include <amptest_main.h>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

int test() __GPU
{
    index<3> ia(5, 5, 5);
    index<3> ib(2, 2, 2);
    index<3> ic(3, 3, 3);
    index<3> ir, it;

    ir = index<3>(4, 4, 4);
    it = ia + ib - ic; // "*" > "+"

    if (!(it == ir))
    {
        return 11;   // test1 scenario1 failed
    }

    ir = index<3>(6, 6, 6);
    it = ia - ib + ic; // "*" > "-"

    if (!(it == ir))
    {
        return 12;   // test1 scenario2 failed
    }

    index<3> i1(1, 1, 1);
    index<3> i2(2, 2, 2);
    index<3> i3(3, 3, 3);
    index<3> i4(4, 4, 4);
    index<3> i5(5, 5, 5);
    index<3> i6(6, 6, 6);
    index<3> i7(7, 7, 7);
    index<3> i8(8, 8, 8);
    index<3> i9(9, 9, 9);

    ir = index<3>(-7, -7, -7);
    it = i1 + i2 + i3 - i4 - i5 + i6 + i7 - i8 - i9 ;

    if (!(it == ir))
    {
        return 13;   // test1 scenario3 failed
    }

    ir = index<3>(19, 19, 19);
    it = (i1 + i2 ) + (i3 - i4 + i5) - i2 + (i6 + i7) - (i8 - i9);

    if (!(it == ir))
    {
        return 14;   // test1 scenario4 failed
    }

    return 0;   // all passed
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

    parallel_for_each(e, [&](index<1> idx) __GPU{
        kernel(idx, result);
    });
    presult = result;

    for (int i = 0; i < size; i++)
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

